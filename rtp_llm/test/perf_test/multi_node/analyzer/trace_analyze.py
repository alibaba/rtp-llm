"""Raw CUDA kernel statistics for one Kineto/Chrome Trace JSON document."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from rtp_llm.test.perf_test.multi_node.analyzer.configs.module_base import (
    normalize_module_config,
)


def analyze_trace(trace, *, min_occurrences=1, module_config=None, setup=None):
    if type(min_occurrences) is not int or min_occurrences <= 0:
        raise ValueError("min_occurrences must be a positive integer")
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise ValueError("expected Chrome Trace JSON with a traceEvents array")
    groups = defaultdict(list)
    timestamps = []
    for event in trace["traceEvents"]:
        if (
            not isinstance(event, dict)
            or event.get("cat") != "kernel"
            or event.get("ph") != "X"
        ):
            continue
        name, start, duration = event.get("name"), event.get("ts"), event.get("dur")
        if not isinstance(name, str) or not name:
            raise ValueError("CUDA kernel event requires a name")
        if (
            any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (start, duration)
            )
            or duration < 0
        ):
            raise ValueError(
                "CUDA kernel timing must be finite with nonnegative duration"
            )
        groups[name].append(duration)
        timestamps.append((start, start + duration))
    if not timestamps:
        raise ValueError("trace contains no complete CUDA kernel events")
    kernels = []
    for name, durations in groups.items():
        if len(durations) < min_occurrences:
            continue
        kernels.append(
            {
                "kernel_name": name,
                "count": len(durations),
                "total_duration_us": sum(durations),
                "avg_duration_us": statistics.mean(durations),
                "min_duration_us": min(durations),
                "max_duration_us": max(durations),
                "median_duration_us": statistics.median(durations),
                "std_dev_us": (
                    statistics.stdev(durations) if len(durations) > 1 else 0.0
                ),
            }
        )
    result = {
        "kernel_count": sum(map(len, groups.values())),
        "kernel_time_us": sum(sum(values) for values in groups.values()),
        "kernel_span_us": max(end for _, end in timestamps)
        - min(start for start, _ in timestamps),
        "kernels": sorted(
            kernels, key=lambda row: (-row["total_duration_us"], row["kernel_name"])
        ),
    }
    if module_config is not None:
        result["modules"] = _module_statistics(
            groups,
            normalize_module_config(module_config),
            setup,
            result["kernel_time_us"],
        )
    return result


def _module_statistics(groups, config, setup, kernel_time):
    modules = []
    for name, spec in config.items():
        matched = {
            key: values for key, values in groups.items() if spec.name_pattern in key
        }
        if not matched:
            continue
        duration = sum(sum(values) for values in matched.values())
        row = {
            "module_name": name,
            "kernel_names": sorted(matched),
            "count": sum(map(len, matched.values())),
            "total_duration_us": duration,
            "kernel_time_percentage": (
                100 * duration / kernel_time if kernel_time else 0.0
            ),
            "estimated_flops_per_invocation": None,
            "estimated_io_bytes_per_invocation": None,
            "estimated_tflops": None,
            "estimated_bandwidth_gbs": None,
            "estimation_error": None,
        }
        formulas = (
            ("estimated_flops_per_invocation", spec.num_flop_calc_func),
            ("estimated_io_bytes_per_invocation", spec.mem_io_calc_func),
        )
        if any(function is not None for _, function in formulas):
            errors = []
            if setup is None or setup.estimation_error:
                errors.append(
                    setup.estimation_error if setup else "run shape context unavailable"
                )
            elif spec.mode is not None and spec.mode != setup.mode:
                errors.append(
                    f"formula targets {spec.mode}, captured mode is {setup.mode}"
                )
            elif spec.tp_size is not None and spec.tp_size != setup.tp_size:
                errors.append(
                    f"formula requires TP={spec.tp_size}, run has TP={setup.tp_size}"
                )
            else:
                for field, function in formulas:
                    if function is None:
                        continue
                    try:
                        value = function(setup)
                        if (
                            type(value) not in (int, float)
                            or not math.isfinite(value)
                            or value < 0
                        ):
                            raise ValueError(
                                "formula must return a finite nonnegative number"
                            )
                        row[field] = value
                    except Exception as error:
                        errors.append(f"{field}: {error}")
                if setup.steps is None:
                    errors.append(
                        "capture phase/step count unavailable; rates not estimated"
                    )
                elif not spec.formula_per_kernel:
                    errors.append(
                        "formula invocation scope is not per kernel; rates not estimated"
                    )
                elif duration <= 0:
                    errors.append("zero captured module duration; rates not estimated")
                else:
                    for field, source, scale in (
                        ("estimated_tflops", "estimated_flops_per_invocation", 1e6),
                        (
                            "estimated_bandwidth_gbs",
                            "estimated_io_bytes_per_invocation",
                            1e3,
                        ),
                    ):
                        if row[source] is not None:
                            row[field] = row[source] * row["count"] / duration / scale
            row["estimation_error"] = "; ".join(errors) or None
        modules.append(row)
    return sorted(
        modules, key=lambda row: (-row["total_duration_us"], row["module_name"])
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--min-occurrences", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = analyze_trace(
            json.loads(args.trace.read_text()), min_occurrences=args.min_occurrences
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"trace analysis failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
