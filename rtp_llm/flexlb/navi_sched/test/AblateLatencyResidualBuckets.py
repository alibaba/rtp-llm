#!/usr/bin/env python3
"""Replay only residual granularity on a validated, fixed-rate C++ base trace.

The input is LatencyOnlineReplay TRACE_MODEL_STATE=1 output and its JSONL
events. Base fitting uses the configured fixed learning rate.
No request scheduling counterfactual is implied by completion-time scoring.
"""

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics

MIN_LOG = 4
MAX_LOG = 22
LN2 = math.log(2.0)


def parse_model_state(line):
    fields = line.rstrip().split("\t")
    try:
        parameter_count = int(fields[24])
        residual_count = int(fields[25 + parameter_count])
        if (fields[0] != "M" or parameter_count != 17 or residual_count != 19 or
                len(fields) != 26 + parameter_count + residual_count):
            raise ValueError("unexpected model-state shape")
        result = {"event_time_us": int(fields[6]), "target_ms": float(fields[7]),
                  "base_ms": float(fields[21]),
                  "residual_parameters": [float(x) for x in fields[26 + parameter_count:]]}
    except (IndexError, ValueError) as error:
        raise ValueError("invalid C++ model-state trace") from error
    if any(not math.isfinite(x) for x in
           [result["target_ms"], result["base_ms"], *result["residual_parameters"]]):
        raise ValueError("nonfinite C++ model-state trace")
    return result


def smoother_step(value):
    return value ** 3 * (value * (6.0 * value - 15.0) + 10.0)


class ResidualGrid:
    def __init__(self, subdivisions, coarse_parameters=None, last_update_us=0,
                 project_updates=True):
        if subdivisions not in (1, 2, 4, 8):
            raise ValueError("subdivisions must be 1, 2, 4 or 8")
        self.subdivisions = subdivisions
        self.parameters = [0.0] * ((MAX_LOG - MIN_LOG) * subdivisions + 1)
        self.last_update_us = last_update_us
        width = 1.0 / subdivisions
        log_width = LN2 * width
        fraction = (log_width + 4.0 - math.sqrt(log_width ** 2 + 16.0)) / (2.0 * log_width)
        maximum = 30.0 * fraction ** 2 * (fraction - 1.0) ** 2 / 2 ** (width * fraction)
        self.limits = [log_width / maximum * 2 ** (MIN_LOG + i * width)
                       for i in range(len(self.parameters) - 1)]
        self.offsets = [0.0]
        for limit in self.limits:
            self.offsets.append(self.offsets[-1] + limit)
        self.update_count = 0
        self.clipped_count = 0
        self.projected_count = 0
        self.project_updates = project_updates
        self.nonmonotonic_count = 0
        if coarse_parameters is not None:
            if len(coarse_parameters) != 19 or any(not math.isfinite(x) for x in coarse_parameters):
                raise ValueError("invalid original residual parameters")
            if subdivisions == 1:
                self.parameters = list(coarse_parameters)
            else:
                coarse = ResidualGrid(1, coarse_parameters)
                self.parameters = [coarse.correction(2 ** (MIN_LOG + i * width), 0)
                                   for i in range(len(self.parameters))]
                self.project()
        self.validate()

    def validate(self):
        if self.parameters[0] != 0 or self.parameters[-1] != 0:
            raise ValueError("endpoint residual must be zero")
        for i, limit in enumerate(self.limits):
            tolerance = 1e-8 * max(1.0, limit, abs(self.parameters[i]))
            if not math.isfinite(self.parameters[i]) or self.parameters[i + 1] - self.parameters[i] < -limit - tolerance:
                raise ValueError("nonmonotonic residual parameters")

    def interpolation(self, base):
        if not math.isfinite(base) or not 2 ** MIN_LOG <= base < 2 ** MAX_LOG:
            return None
        position = (math.log2(base) - MIN_LOG) * self.subdivisions
        lower = min(len(self.parameters) - 2, math.floor(position))
        upper_weight = smoother_step(max(0.0, min(1.0, position - lower)))
        return lower, 1.0 - upper_weight, upper_weight

    def decay(self, now_us):
        if self.last_update_us <= 0 or now_us <= self.last_update_us:
            return 1.0
        idle = now_us - self.last_update_us
        return max(0.0, min(1.0, (300_000_000 - idle) / 180_000_000))

    def correction(self, base, now_us):
        weights = self.interpolation(base)
        if weights is None:
            return 0.0
        index, lower, upper = weights
        return self.decay(now_us) * (lower * self.parameters[index] + upper * self.parameters[index + 1])

    def predict(self, base, now_us):
        return max(0.0, base + self.correction(base, now_us))

    def project(self):
        # PAVA on offset parameters enforces the production monotonicity bound.
        blocks = []
        for i in range(1, len(self.parameters) - 1):
            value = max(0.0, min(self.offsets[-1], self.parameters[i] + self.offsets[i]))
            blocks.append([i, i + 1, value, 1])
            while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
                right = blocks.pop()
                blocks[-1][1] = right[1]
                blocks[-1][2] += right[2]
                blocks[-1][3] += right[3]
        for begin, end, total, count in blocks:
            value = total / count
            for i in range(begin, end):
                self.parameters[i] = value - self.offsets[i]

    def update(self, base, actual, now_us, rate=0.8, floor_ms=100.0, ratio=0.1):
        if not (math.isfinite(actual) and actual >= 0 and 0 < rate <= 1 and floor_ms >= 0 and ratio >= 0):
            raise ValueError("invalid update")
        weights = self.interpolation(base)
        if weights is None:
            return
        index, lower, upper = weights
        now_us = max(now_us, self.last_update_us)
        factor = self.decay(now_us)
        self.parameters = [value * factor for value in self.parameters]
        correction = lower * self.parameters[index] + upper * self.parameters[index + 1]
        error = actual - base - correction
        limit = max(floor_ms, ratio * base)
        self.clipped_count += abs(error) > limit
        error = max(-limit, min(limit, error))
        denominator = lower ** 2 + upper ** 2
        if index != 0:
            self.parameters[index] += rate * error * lower / denominator
        if index + 1 != len(self.parameters) - 1:
            self.parameters[index + 1] += rate * error * upper / denominator
        before_projection = self.parameters[:]
        if self.project_updates:
            self.project()
        self.projected_count += max(abs(a - b) for a, b in zip(before_projection, self.parameters)) > 1e-6
        self.nonmonotonic_count += any(self.parameters[i + 1] - self.parameters[i] < -limit - 1e-6
                                      for i, limit in enumerate(self.limits))
        self.last_update_us = now_us
        self.update_count += 1


def metrics(rows):
    if not rows:
        return {"n": 0}
    errors = [predicted - actual for actual, predicted in rows]
    ordered = sorted(abs(x) for x in errors)
    def percentile(values):
        values = sorted(values)
        return values[math.ceil(0.95 * len(values)) - 1]
    return {
        "n": len(rows),
        "mae_ms": statistics.mean(ordered),
        "bias_ms": statistics.mean(errors),
        "wape_pct": 100 * sum(ordered) / max(1e-12, sum(x[0] for x in rows)),
        "p95_abs_ms": percentile(ordered),
        "p95_over_ms": percentile([max(0.0, x) for x in errors]),
        "p95_under_ms": percentile([max(0.0, -x) for x in errors]),
    }


def phase_labels(samples):
    buckets = collections.defaultdict(list)
    for sample in samples.values():
        ready_us = sample["event_time_us"] - round(sample["target_ms"] * 1000)
        buckets[ready_us // 30_000_000].append(sample["features"][6])
    means = {key: statistics.mean(values) for key, values in buckets.items()}
    result = {}
    for sequence, sample in samples.items():
        ready_us = sample["event_time_us"] - round(sample["target_ms"] * 1000)
        key = ready_us // 30_000_000
        if key - 1 not in means:
            result[sequence] = "unknown"
            continue
        delta = means[key] - means[key - 1]
        threshold = max(4.0, 0.1 * means[key - 1])
        result[sequence] = "rising" if delta > threshold else "falling" if delta < -threshold else "turn"
    return result


def run_host(job):
    host, architecture, lines, event_path, options = job
    samples = {}
    first_update = None
    with Path(event_path).open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["type"] == "sample":
                samples[event["sequence"]] = event
            elif first_update is None:
                first_update = event
    if first_update is None:
        raise ValueError("missing initial state: " + host)
    phases = phase_labels(samples)
    original = first_update["residual_parameters_before"]
    initial_time = first_update["residual_time_before"]
    grids = {str(d): ResidualGrid(d, original, initial_time) for d in options["subdivisions"]}
    grids.update({"zero_" + str(d): ResidualGrid(d) for d in options["subdivisions"]})
    if options.get("projection_control"):
        grids.update({"no_projection_" + str(d): ResidualGrid(d, original, initial_time, False)
                      for d in options["subdivisions"]})
    checked = collections.Counter()
    differences = collections.defaultdict(float)
    groups = {key: collections.defaultdict(list) for key in grids}
    zero_points = {key: [] for key in grids}
    initialization_max = {key: 0.0 for key in grids}
    for i in range(1001):
        base = 2 ** (MIN_LOG + (MAX_LOG - MIN_LOG) * i / 1000)
        for key, grid in grids.items():
            initialization_max[key] = max(initialization_max[key], abs(grid.predict(base, initial_time) - grids["1"].predict(base, initial_time)))
    for line in lines:
        fields = line.rstrip().split("\t")
        if fields[0] == "M":
            state = parse_model_state(line)
            for grid in grids.values():
                grid.update(state["base_ms"], state["target_ms"], state["event_time_us"],
                            options["rate"], options["floor_ms"], options["ratio"])
            delta = max(abs(a - b) for a, b in zip(grids["1"].parameters, state["residual_parameters"]))
            differences["cpp_state_max_ms"] = max(differences["cpp_state_max_ms"], delta)
            checked["updates"] += 1
        elif fields[0] == "P":
            sequence, now_us = int(fields[4]), int(fields[5])
            actual, base, cpp_prediction = float(fields[6]), float(fields[8]), float(fields[9])
            logged_base, logged_prediction = float(fields[11]), float(fields[12])
            if any(not math.isfinite(x) for x in (actual, base, cpp_prediction, logged_base, logged_prediction)):
                raise ValueError("nonfinite C++ prediction: " + host)
            differences["online_base_max_ms"] = max(differences["online_base_max_ms"], abs(base - logged_base))
            differences["online_final_max_ms"] = max(differences["online_final_max_ms"], abs(cpp_prediction - logged_prediction))
            differences["cpp_prediction_max_ms"] = max(differences["cpp_prediction_max_ms"], abs(grids["1"].predict(base, now_us) - cpp_prediction))
            checked["predictions"] += 1
            if not options["start_us"] <= now_us < options["end_us"]:
                continue
            sample = samples[sequence]
            running, waiting = sample["features"][1], sample["features"][6]
            pressure = "low" if running <= 4 and waiting <= 4 else "high" if waiting >= 33 else "middle"
            settled = now_us >= options["start_us"] + 300_000_000
            segments = ["all", pressure, phases[sequence], pressure + "_" + phases[sequence],
                        "after5min" if settled else "first5min"]
            if settled:
                segments.extend(["after5min_" + pressure, "after5min_" + phases[sequence]])
            if actual < 256:
                segments.append("short_lt256ms")
            for key, grid in grids.items():
                pair = (actual, grid.predict(base, now_us))
                for segment in segments:
                    groups[key][segment].append(pair)
        elif fields[0] == "Z":
            now_us, base = int(fields[5]), float(fields[6])
            if options["start_us"] <= now_us < options["end_us"]:
                for key, grid in grids.items():
                    zero_points[key].append(grid.predict(base, now_us))
        elif fields[0] == "U":
            if any(not math.isfinite(float(x)) for x in fields[6:11]):
                raise ValueError("nonfinite C++ update: " + host)
            differences["online_parameters_max"] = max(differences["online_parameters_max"], float(fields[6]))
            differences["online_residual_state_max_ms"] = max(differences["online_residual_state_max_ms"], float(fields[7]))
            if int(fields[10]) != 0:
                raise ValueError("residual timestamp mismatch: " + host)
    if not checked["predictions"] or max(differences.values(), default=0) > options["tolerance"]:
        raise ValueError("baseline did not reproduce online/C++: " + host + " " + str(dict(differences)))
    result = {"host": host, "architecture": architecture, "checks": dict(checked),
              "differences": dict(differences), "initialization_max_prediction_difference_ms": initialization_max,
              "variants": {key: {"metrics": {segment: metrics(rows) for segment, rows in group.items()},
                                    "updates": grids[key].update_count, "clipped": grids[key].clipped_count,
                                    "projected": grids[key].projected_count,
                                    "nonmonotonic_updates": grids[key].nonmonotonic_count,
                                    "zero_point_min_mean_max_ms": [min(zero_points[key]), statistics.mean(zero_points[key]), max(zero_points[key])]}
                           for key, group in groups.items()}}
    return result, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--end-us", type=int, required=True)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rate", type=float, default=0.8)
    parser.add_argument("--floor-ms", type=float, default=100.0)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--projection-control", action="store_true",
                        help="Also disable monotonic projection for diagnosis only; not a deployable variant")
    args = parser.parse_args()
    if 1 not in args.subdivisions or args.end_us <= args.start_us or args.workers < 1:
        parser.error("require baseline subdivision 1, positive workers and a nonempty interval")
    if args.trace.stat().st_size > 1024 ** 3:
        parser.error("trace exceeds the 1 GiB input limit")
    manifest = json.loads(args.manifest.read_text())
    architectures = {}
    for node in manifest["nodes"]:
        deployment = node["deployment"]
        if "MI308" in deployment or "RTX_PRO_5000" in deployment:
            architectures[node["ip"].replace(".", "_")] = "MI308" if "MI308" in deployment else "RTX5000"
    traces = collections.defaultdict(list)
    with args.trace.open() as stream:
        for line in stream:
            fields = line.split("\t", 2)
            if len(fields) >= 3 and fields[0] in ("M", "P", "U", "Z"):
                traces[fields[1]].append(line)
    options = vars(args)
    jobs = [(host, architectures[host], lines, str(args.events / (host + ".jsonl")), options)
            for host, lines in traces.items()]
    combined = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    hosts = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        for result, groups in executor.map(run_host, jobs):
            hosts.append(result)
            for variant, segments in groups.items():
                for segment, rows in segments.items():
                    combined[result["architecture"]][variant][segment].extend(rows)
    print(json.dumps({
        "definitions": {"scoring": "Agent completion-time prediction before fitting the sample; not Master admission",
                        "base_trajectory": "fixed C++ trace, verified against online parameters and predictions",
                        "bias": "mean(prediction-actual); positive is overprediction",
                        "wape_pct": "100*sum(abs(prediction-actual))/sum(actual)",
                        "pressure": "low: running<=4 and waiting<=4; high: waiting>=33; otherwise middle",
                        "phase": "per-node waiting-count mean in 30s inferred-ready bins versus previous bin; threshold max(4,10% previous)",
                        "ready_time": "completion_time-target_ms*1000; phase is descriptive, not a predictor",
                        "zero_prefix": "all residual parameters reset to zero at the same anchor, as an initialization sensitivity check",
                        "no_projection_prefix": "diagnostic only: disable monotonic projection, allowing invalid production curves",
                        "p95_over_under": "95th percentile of max(error,0) and max(-error,0), including zeros"},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in options.items()},
        "hosts": hosts,
        "aggregate": {arch: {variant: {segment: metrics(rows) for segment, rows in segments.items()}
                             for variant, segments in variants.items()} for arch, variants in combined.items()},
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
