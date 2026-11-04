#!/usr/bin/env python3
"""Summarize per-parameter Adam update traces by hardware and traffic phase."""

import argparse
import json
import math
from array import array
from collections import defaultdict
from pathlib import Path


PARAMETER_COUNT = 19
PARAMETER_SCALES = (
    0.878862443703, 0.119054644545, 0.0881092493705,
    2.45290259606, 0.918359382716, 19.5646708383, 1.0,
    1.23972355201, 0.00229934153041, 0.211011741932,
    1.09757235126, 0.878792874576, 193.929503215,
    222.305482002, 207.484099949, 0.0242867645882,
    1892.09311928, 0.0299401936353, 0.403880546944,
)
TRACE_PREFIX_FIELDS = 14
TRACE_PARAMETER_FIELDS = 15
METRIC_OFFSETS = {
    "raw_gradient": 0,
    "normalized_gradient": 7,
    "parameter_before": 8,
    "parameter_after": 10,
    "candidate_update": 11,
    "actual_update": 12,
    "prediction_shift_ms": 14,
}
DERIVED_METRICS = (
    "effective_coefficient_gradient",
    "effective_coefficient_update",
    "prediction_sensitivity_ms_per_coordinate",
    "prediction_sensitivity_ms_per_effective_coefficient",
)


def load_host_architecture(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for node in manifest.get("nodes", []):
        host = node["ip"].replace(".", "_")
        deployment = node["deployment"].upper()
        if "MI308" in deployment:
            result[host] = "MI308"
            result[f"{node['deployment']}_{host}"] = "MI308"
        elif "RTX_PRO_5000" in deployment:
            result[host] = "RTX5000"
            result[f"{node['deployment']}_{host}"] = "RTX5000"
    if not result:
        raise ValueError("manifest contains no MI308 or RTX5000 nodes")
    return result


def parse_host_architecture(value):
    try:
        host, architecture = value.rsplit("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "host architecture must be HOST=MI308 or HOST=RTX5000"
        ) from error
    host = host.replace(".", "_")
    if not host or architecture not in ("MI308", "RTX5000"):
        raise argparse.ArgumentTypeError(
            "host architecture must be HOST=MI308 or HOST=RTX5000"
        )
    return host, architecture


def classify_sine_phase(ready_time_us, start_time_us, period_us):
    if start_time_us is None or ready_time_us < start_time_us:
        return "unknown"
    position = ((ready_time_us - start_time_us) % period_us) / period_us
    if position == 0.0 or position == 0.5:
        return "turning"
    return "rising" if position < 0.5 else "falling"


def load_pressure_means(trace_path, host_architecture, bin_us):
    pressure_sum = defaultdict(float)
    pressure_count = defaultdict(int)
    with trace_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.startswith("M\t"):
                continue
            fields = line.rstrip("\n").split("\t")
            try:
                host = fields[1]
                event_time_us = int(fields[6])
                target_ms = float(fields[7])
                running = float(fields[9])
                waiting = float(fields[14])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"{trace_path}:{line_number}: invalid model-state line"
                ) from error
            architecture = host_architecture.get(host)
            if architecture is None:
                raise ValueError(f"trace host is absent from manifest: {host}")
            ready_time_us = event_time_us - round(target_ms * 1000.0)
            key = (architecture, ready_time_us // bin_us)
            pressure_sum[key] += running + waiting
            pressure_count[key] += 1
    if not pressure_count:
        raise ValueError("trace contains no model-state pressure records")
    return {
        key: pressure_sum[key] / count
        for key, count in pressure_count.items()
    }


def classify_pressure_phase(architecture, ready_time_us, pressure_means,
                            bin_us):
    current_bin = ready_time_us // bin_us
    current = pressure_means.get((architecture, current_bin))
    if current is None:
        return "unknown"
    previous = pressure_means.get((architecture, current_bin - 1), current)
    return "rising" if current >= previous else "falling"


def parse_trace_line(line):
    fields = line.rstrip("\n").split("\t")
    if not fields or fields[0] != "D":
        return None
    try:
        parameter_count = int(fields[13])
        expected_fields = TRACE_PREFIX_FIELDS + (
            parameter_count * TRACE_PARAMETER_FIELDS
        )
        if parameter_count != PARAMETER_COUNT or len(fields) != expected_fields:
            raise ValueError("unexpected parameter trace shape")
        values = {
            metric: tuple(
                float(fields[TRACE_PREFIX_FIELDS + index * TRACE_PARAMETER_FIELDS + offset])
                for index in range(parameter_count)
            )
            for metric, offset in METRIC_OFFSETS.items()
        }
        return {
            "host": fields[1],
            "event_time_us": int(fields[6]),
            "target_ms": float(fields[7]),
            "parameter_count": parameter_count,
            "values": values,
        }
    except (IndexError, ValueError) as error:
        raise ValueError("invalid Adam trace line") from error


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(values):
    if not values:
        return {"count": 0}
    raw = list(values)
    absolute = sorted(abs(value) for value in raw)
    count = len(raw)
    return {
        "count": count,
        "signed_mean": sum(raw) / count,
        "absolute_mean": sum(absolute) / count,
        "absolute_p50": percentile(absolute, 0.50),
        "absolute_p95": percentile(absolute, 0.95),
        "rms": math.sqrt(sum(value * value for value in raw) / count),
        "maximum_absolute": absolute[-1],
    }


def analyze(trace_path, host_architecture, sine_start_us=None,
            sine_period_us=300_000_000, window_start_us=None,
            window_end_us=None, phase_mode="sine",
            pressure_bin_us=30_000_000):
    if phase_mode not in ("sine", "pressure-trend"):
        raise ValueError("invalid phase mode")
    pressure_means = (
        load_pressure_means(trace_path, host_architecture, pressure_bin_us)
        if phase_mode == "pressure-trend" else None
    )
    values = defaultdict(lambda: array("d"))
    update_counts = defaultdict(int)
    with trace_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.startswith("D\t"):
                continue
            try:
                record = parse_trace_line(line)
            except ValueError as error:
                raise ValueError(f"{trace_path}:{line_number}: {error}") from error
            ready_time_us = record["event_time_us"] - round(
                record["target_ms"] * 1000.0
            )
            if window_start_us is not None and ready_time_us < window_start_us:
                continue
            if window_end_us is not None and ready_time_us >= window_end_us:
                continue
            architecture = host_architecture.get(record["host"])
            if architecture is None:
                raise ValueError(f"trace host is absent from manifest: {record['host']}")
            phase = (
                classify_pressure_phase(
                    architecture, ready_time_us, pressure_means,
                    pressure_bin_us,
                )
                if pressure_means is not None
                else classify_sine_phase(
                    ready_time_us, sine_start_us, sine_period_us
                )
            )
            for group_phase in ("all", phase):
                update_counts[(architecture, group_phase)] += 1
                for metric, metric_values in record["values"].items():
                    for index, value in enumerate(metric_values):
                        if not math.isfinite(value):
                            raise ValueError(
                                f"non-finite {metric} for {record['host']} P{index}"
                            )
                        values[(architecture, group_phase, metric, index)].append(value)
                updates = record["values"]["actual_update"]
                gradients = record["values"]["raw_gradient"]
                shifts = record["values"]["prediction_shift_ms"]
                for index, (gradient, update, shift) in enumerate(
                        zip(gradients, updates, shifts)):
                    scale = PARAMETER_SCALES[index]
                    values[(architecture, group_phase,
                            "effective_coefficient_gradient", index)].append(
                                gradient / scale
                            )
                    values[(architecture, group_phase,
                            "effective_coefficient_update", index)].append(
                                update * scale
                            )
                    if abs(update) <= 1e-12:
                        continue
                    coordinate_sensitivity = shift / update
                    effective_sensitivity = coordinate_sensitivity / scale
                    if (not math.isfinite(coordinate_sensitivity)
                            or not math.isfinite(effective_sensitivity)):
                        raise ValueError(
                            f"non-finite prediction sensitivity for "
                            f"{record['host']} P{index}"
                        )
                    values[(architecture, group_phase,
                            "prediction_sensitivity_ms_per_coordinate",
                            index)].append(coordinate_sensitivity)
                    values[(architecture, group_phase,
                            "prediction_sensitivity_ms_per_effective_coefficient",
                            index)].append(effective_sensitivity)

    result = {"architectures": {}}
    for architecture in sorted(set(host_architecture.values())):
        phases = {}
        for phase in ("all", "rising", "falling", "turning", "unknown"):
            count = update_counts[(architecture, phase)]
            if count == 0:
                continue
            metrics = {}
            for metric in (*METRIC_OFFSETS, *DERIVED_METRICS):
                metrics[metric] = [
                    summarize(values[(architecture, phase, metric, index)])
                    for index in range(PARAMETER_COUNT)
                ]
            phases[phase] = {"update_count": count, "metrics": metrics}
        result["architectures"][architecture] = {"phases": phases}
    return result


def print_summary(result):
    columns = (
        ("raw_gradient", "rms"),
        ("effective_coefficient_gradient", "rms"),
        ("normalized_gradient", "absolute_p50"),
        ("actual_update", "absolute_p50"),
        ("effective_coefficient_update", "absolute_p50"),
        ("prediction_shift_ms", "absolute_p50"),
        ("prediction_shift_ms", "absolute_p95"),
        ("prediction_sensitivity_ms_per_effective_coefficient", "absolute_p50"),
        ("prediction_sensitivity_ms_per_effective_coefficient", "absolute_p95"),
    )
    for architecture, architecture_data in result["architectures"].items():
        for phase, phase_data in architecture_data["phases"].items():
            print(f"{architecture} {phase} updates={phase_data['update_count']}")
            print("P coordinate_grad_rms effective_grad_rms "
                  "normalized_abs_p50 coordinate_update_abs_p50 "
                  "effective_update_abs_p50 shift_abs_p50_ms "
                  "shift_abs_p95_ms effective_sensitivity_abs_p50 "
                  "effective_sensitivity_abs_p95")
            for index in range(PARAMETER_COUNT):
                row = []
                for metric, statistic in columns:
                    row.append(phase_data["metrics"][metric][index].get(statistic))
                print("{} {}".format(
                    index,
                    " ".join(
                        "n/a" if value is None else f"{value:.9g}"
                        for value in row
                    ),
                ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sine-start-us", type=int)
    parser.add_argument("--sine-period-seconds", type=float, default=300.0)
    parser.add_argument("--window-start-us", type=int)
    parser.add_argument("--window-end-us", type=int)
    parser.add_argument("--phase-mode", choices=("sine", "pressure-trend"),
                        default="sine")
    parser.add_argument("--pressure-bin-seconds", type=float, default=30.0)
    parser.add_argument(
        "--host-architecture",
        action="append",
        default=[],
        type=parse_host_architecture,
        metavar="HOST=ARCH",
        help="override a missing/replaced trace host with MI308 or RTX5000",
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.sine_period_seconds)
        or args.sine_period_seconds <= 0.0
        or not math.isfinite(args.pressure_bin_seconds)
        or args.pressure_bin_seconds <= 0.0
        or (
            args.window_start_us is not None
            and args.window_end_us is not None
            and args.window_start_us >= args.window_end_us
        )
    ):
        parser.error("invalid sine period or analysis window")
    host_architecture = load_host_architecture(args.manifest)
    host_architecture.update(args.host_architecture)
    result = analyze(
        args.trace,
        host_architecture,
        args.sine_start_us,
        round(args.sine_period_seconds * 1_000_000),
        args.window_start_us,
        args.window_end_us,
        args.phase_mode,
        round(args.pressure_bin_seconds * 1_000_000),
    )
    if args.output:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print_summary(result)


if __name__ == "__main__":
    main()
