#!/usr/bin/env python3
"""Compare explicitly anchored frozen fitting states with online replay."""

import argparse
import collections
import json
import math
from pathlib import Path
import subprocess
import tempfile


def load_events(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def select_anchors(events, start_us):
    samples = [event for event in events
               if event["type"] == "sample" and event["event_time_us"] >= start_us]
    updates = [event for event in events if event["type"] == "update"]
    high = next(event for event in samples if event["target_ms"] >= 20_000)
    preceding = [event["sequence"] for event in updates
                 if event["sequence"] < high["sequence"]]
    low_anchor = max(preceding) if preceding else None
    high_anchor = min(event["sequence"] for event in updates
                      if event["sequence"] >= high["sequence"])
    return high, low_anchor, high_anchor


def select_sample_pressure_anchor(events, before_us, maximum_pressure):
    samples = {event.get("history_sequence", event["sequence"]): event
               for event in events if event["type"] == "sample"}
    eligible = []
    for event in events:
        if event["type"] != "update" or not event.get("fit_batches"):
            continue
        current_sequence = event["fit_batches"][-1][-1]
        current = samples.get(current_sequence)
        if current is None or len(current.get("features", [])) < 7:
            raise ValueError("missing current fitting sample pressure")
        if (current["event_time_us"] <= before_us and
                current["features"][1] + current["features"][6] <= maximum_pressure):
            eligible.append((current["event_time_us"], event["sequence"]))
    if not eligible:
        raise ValueError("no fitting update at the requested sample pressure")
    return max(eligible)[1]


def read_predictions(path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if fields[0] == "P":
                rows[(fields[1], int(fields[4]))] = (
                    int(fields[5]), float(fields[6]), float(fields[9]),
                    float(fields[12]),
                )
    return rows


def metrics(rows):
    if not rows:
        return None
    online_errors = [current - actual for actual, _, current, _ in rows]
    frozen_errors = [frozen_prediction - actual
                     for actual, frozen_prediction, _, _ in rows]
    online = sum(map(abs, online_errors))
    frozen = sum(map(abs, frozen_errors))
    actual = sum(actual for actual, _, _, _ in rows)
    def p95(values):
        values = sorted(values)
        return values[math.ceil(0.95 * len(values)) - 1] if values else 0.0
    return {"count": len(rows), "online_mae_ms": online / len(rows),
            "frozen_mae_ms": frozen / len(rows),
            "online_wape": online / actual, "frozen_wape": frozen / actual,
            "online_gain_fraction": (frozen - online) / frozen,
            "online_bias_ms": sum(online_errors) / len(rows),
            "frozen_bias_ms": sum(frozen_errors) / len(rows),
            "online_over_p95_ms": p95([max(0.0, error) for error in online_errors]),
            "frozen_over_p95_ms": p95([max(0.0, error) for error in frozen_errors]),
            "online_under_p95_ms": p95([max(0.0, -error) for error in online_errors]),
            "frozen_under_p95_ms": p95([max(0.0, -error) for error in frozen_errors])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--end-us", type=int, required=True)
    parser.add_argument("--high-after-us", type=int,
                        help="Select the next sample with target >=20s after this time")
    parser.add_argument("--anchor-before-us", type=int,
                        help="Freeze at a fitting update no later than this sample event time")
    parser.add_argument("--max-anchor-sample-pressure", type=int,
                        help="Maximum current training sample running+waiting count")
    parser.add_argument("--scale", type=float, default=16.0)
    parser.add_argument("--residual", type=float, default=0.8)
    parser.add_argument("--modes", nargs="+",
                        choices=("freeze", "freeze_base", "freeze_residual"),
                        default=["freeze"])
    args = parser.parse_args()
    if ((args.anchor_before_us is None) !=
            (args.max_anchor_sample_pressure is None)):
        parser.error("sample-pressure anchor requires both time and pressure")
    if args.max_anchor_sample_pressure is not None and args.max_anchor_sample_pressure < 0:
        parser.error("sample-pressure anchor must be nonnegative")
    baseline = read_predictions(args.baseline)
    aggregate = collections.defaultdict(list)
    for path in sorted(args.events.glob("*.jsonl")):
        events = load_events(path)
        high, low_anchor, high_anchor = select_anchors(
            events, args.high_after_us or args.start_us)
        high_time = high["event_time_us"]
        architecture = "MI" if path.stem.startswith("mi-") else "RTX"
        if args.anchor_before_us is not None:
            anchor = select_sample_pressure_anchor(
                events, args.anchor_before_us, args.max_anchor_sample_pressure)
            if anchor >= high["sequence"]:
                raise ValueError("sample-pressure anchor must precede evaluated high sample")
            anchors = [("sample_pressure", anchor)]
        else:
            anchors = [("after_first_high", high_anchor)]
            if low_anchor is not None:
                anchors.insert(0, ("before_high", low_anchor))
        for label, anchor in anchors:
            for mode in args.modes:
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    with (directory / path.name).open("w", encoding="utf-8") as stream:
                        for event in events:
                            if event["type"] == "sample" or event["sequence"] >= anchor:
                                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
                    trace = directory / "frozen.tsv"
                    completed = subprocess.run(
                        [str(args.binary), str(directory), str(trace), "4", "20261008",
                         str(args.scale), str(args.residual), "all", "0", mode],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        universal_newlines=True,
                    )
                    if completed.returncode:
                        raise RuntimeError(completed.stderr.strip())
                    frozen = read_predictions(trace)
                groups = {"all_high": [], "first_10min": [], "after_10min": []}
                for key, (time_us, actual, predicted, _) in frozen.items():
                    if not high_time <= time_us < args.end_us or actual < 20_000:
                        continue
                    old = baseline.get(key)
                    if old is None or abs(old[2] - old[3]) > 0.1:
                        continue
                    row = (actual, predicted, old[2], time_us)
                    groups["all_high"].append(row)
                    segment = "first_10min" if time_us < high_time + 600_000_000 else "after_10min"
                    groups[segment].append(row)
                for segment, rows in groups.items():
                    aggregate[(architecture, label, mode, segment)].extend(rows)
                print(json.dumps({"host": path.stem, "architecture": architecture,
                                  "anchor": label, "mode": mode,
                                  "anchor_sequence": anchor,
                                  "high_start_us": high_time,
                                  "first_high_target_ms": high["target_ms"],
                                  "first_high_prediction_ms": high["prediction_before_ms"],
                                  "segments": {name: metrics(rows)
                                               for name, rows in groups.items()}},
                                 ensure_ascii=False))
    for (architecture, label, mode, segment), rows in sorted(aggregate.items()):
        print(json.dumps({"architecture": architecture, "anchor": label,
                          "mode": mode, "segment": segment,
                          "aggregate": metrics(rows)},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
