#!/usr/bin/env python3
"""Audit residual update clipping from latency_online_replay model-state traces."""

import argparse
import collections
import math
import statistics


MIN_BUCKET = 4
MAX_BUCKET = 22
RESIDUAL_COUNT = MAX_BUCKET - MIN_BUCKET + 1
IDLE_DECAY_START_US = 120_000_000
IDLE_DECAY_END_US = 300_000_000


def residual_correction(base_ms, parameters, last_update_us, event_time_us):
    if base_ms < 2 ** MIN_BUCKET or base_ms > 2 ** MAX_BUCKET:
        return 0.0
    position = math.log2(base_ms)
    lower_bucket = max(MIN_BUCKET, math.floor(position))
    if lower_bucket >= MAX_BUCKET:
        return 0.0
    fraction = min(1.0, max(0.0, position - lower_bucket))
    upper_weight = fraction ** 3 * (fraction * (fraction * 6 - 15) + 10)
    index = lower_bucket - MIN_BUCKET
    correction = (parameters[index] * (1 - upper_weight) +
                  parameters[index + 1] * upper_weight)
    if last_update_us > 0 and event_time_us > last_update_us:
        idle_us = event_time_us - last_update_us
        if idle_us >= IDLE_DECAY_END_US:
            return 0.0
        if idle_us > IDLE_DECAY_START_US:
            correction *= (IDLE_DECAY_END_US - idle_us) / (
                IDLE_DECAY_END_US - IDLE_DECAY_START_US)
    return correction


def parse_model_state(line):
    fields = line.rstrip("\n").split("\t")
    if fields[0] != "M":
        return None
    try:
        parameter_count = int(fields[24])
        residual_count = int(fields[25 + parameter_count])
    except (IndexError, ValueError) as error:
        raise ValueError("invalid model-state trace") from error
    if (parameter_count not in (10, 17) or residual_count != RESIDUAL_COUNT or
            len(fields) != 26 + parameter_count + residual_count):
        raise ValueError("unexpected model-state trace shape")
    return {
        "host": fields[1],
        "sequence": int(fields[4]),
        "event_time_us": int(fields[6]),
        "target_ms": float(fields[7]),
        "running_count": int(fields[9]),
        "waiting_count": int(fields[14]),
        "base_ms": float(fields[21]),
        "logged_correction_ms": float(fields[23]),
        "residual_parameters": tuple(float(value) for value in
                                     fields[26 + parameter_count:]),
    }


def audit(path):
    previous = {}
    groups = collections.defaultdict(list)
    checked = 0
    mismatched = 0
    max_correction_difference = 0.0
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("M\t"):
                continue
            state = parse_model_state(line)
            host = state["host"]
            base = state["base_ms"]
            event_time_us = state["event_time_us"]
            previous_state = previous.get(host)
            last_update_us = previous_state[1] if previous_state else 0
            if base >= 2 ** MIN_BUCKET:
                last_update_us = max(last_update_us, event_time_us)
            actual_correction = residual_correction(
                base, state["residual_parameters"], last_update_us,
                event_time_us)
            difference = abs(actual_correction - state["logged_correction_ms"])
            max_correction_difference = max(max_correction_difference, difference)
            checked += 1
            if difference > 1e-6 * max(1.0, abs(state["logged_correction_ms"])):
                mismatched += 1
            if (previous_state and
                    0 <= state["sequence"] - previous_state[2] <= 16):
                before_correction = residual_correction(
                    base, previous_state[0], previous_state[1], event_time_us)
                innovation = state["target_ms"] - base - before_correction
                limit = max(100.0, 0.1 * base)
                architecture = ("MI" if "MI308" in host or host.startswith("mi-")
                                else "RTX")
                pressure = ("low" if state["running_count"] <= 1 and
                            state["waiting_count"] == 0 else
                            "high" if state["running_count"] >= 9 else "mid")
                observation = (abs(innovation) > limit, abs(innovation) / limit,
                               abs(innovation), limit)
                groups[(architecture, "all")].append(observation)
                groups[(architecture, pressure)].append(observation)
            previous[host] = (state["residual_parameters"], last_update_us,
                              state["sequence"])
    return groups, checked, mismatched, max_correction_difference


def report(label, result):
    groups, checked, mismatched, max_difference = result
    print(label, "state_checks", checked, "mismatches", mismatched,
          "max_correction_difference_ms", round(max_difference, 9))
    for architecture in ("MI", "RTX"):
        for pressure in ("all", "low", "mid", "high"):
            observations = groups[(architecture, pressure)]
            if not observations:
                continue
            print(label, architecture, pressure, "updates", len(observations),
                  "clipped_pct", round(100 * sum(x[0] for x in observations) /
                                       len(observations), 3),
                  "mean_abs_innovation_ms",
                  round(statistics.mean(x[2] for x in observations), 3),
                  "mean_limit_ms",
                  round(statistics.mean(x[3] for x in observations), 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", metavar="LABEL=TRACE")
    args = parser.parse_args()
    for argument in args.trace:
        if "=" not in argument:
            parser.error("trace must be LABEL=TRACE")
        label, path = argument.split("=", 1)
        result = audit(path)
        report(label, result)
        if result[2]:
            raise ValueError("residual correction reconstruction did not match trace")


if __name__ == "__main__":
    main()
