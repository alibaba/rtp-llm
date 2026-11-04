#!/usr/bin/env python3
"""Create bounded replay traces with extra fits on recent rising-pressure samples."""

import argparse
import json
import os
from collections import deque
from pathlib import Path


WINDOW_US = 30_000_000


def rising_pressure(samples, now, minimum_per_window):
    previous = [waiting for timestamp, waiting in samples
                if now - 2 * WINDOW_US <= timestamp < now - WINDOW_US]
    recent = [waiting for timestamp, waiting in samples
              if now - WINDOW_US <= timestamp <= now]
    if len(previous) < minimum_per_window or len(recent) < minimum_per_window:
        return False
    return (recent[-1] >= 33 and
            sum(recent) / len(recent) - sum(previous) / len(previous) >= 4)


def transform_file(source, destination, recent_batches, repeat_count,
                   max_output_bytes, minimum_per_window=8,
                   selection="rising", mode="extra_updates"):
    observed = deque()
    known_history = set()
    last_time = None
    triggers = []
    output_bytes = 0
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with source.open(encoding="utf-8") as input_stream, temporary.open(
                "w", encoding="utf-8") as output_stream:
            for line in input_stream:
                event = json.loads(line)
                if event["type"] == "sample":
                    last_time = event["event_time_us"]
                    observed.append((last_time, event["features"][6]))
                    while observed and observed[0][0] < last_time - 2 * WINDOW_US:
                        observed.popleft()
                    if "history_sequence" in event:
                        known_history.add(event["history_sequence"])
                elif event["type"] == "update":
                    batches = event.get("fit_batches", [])
                    if len(batches) != event["step_after"] - event["step_before"]:
                        raise ValueError(f"incomplete exact batches: {source}")
                    rising = (last_time is not None and rising_pressure(
                        observed, last_time, minimum_per_window))
                    selected_pressure = (rising if selection == "rising" else
                                         last_time is not None and observed[-1][1] >= 33
                                         and not rising)
                    if batches and selected_pressure:
                        selected = batches[-recent_batches:]
                        if any(sequence not in known_history for batch in selected
                               for sequence in batch):
                            raise ValueError(f"unobserved fit sample: {source}")
                        if mode == "extra_updates":
                            extras = selected * repeat_count
                            event["fit_batches"] = batches + extras
                            event["step_after"] += len(extras)
                            extra_updates = len(extras)
                        else:
                            event["fit_batches"] = (
                                batches[:-len(selected)] +
                                [batch[:-1] + [batch[-1]] * (repeat_count + 1)
                                 for batch in selected])
                            extra_updates = 0
                        triggers.append({"time_us": last_time,
                                         "extra_updates": extra_updates})
                else:
                    raise ValueError(f"unknown event type: {source}")
                encoded = json.dumps(event, separators=(",", ":")) + "\n"
                output_bytes += len(encoded.encode("utf-8"))
                if output_bytes > max_output_bytes:
                    raise ValueError(f"output capacity exceeded: {source}")
                output_stream.write(encoded)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"host": source.stem, "triggers": triggers,
            "extra_updates": sum(item["extra_updates"] for item in triggers),
            "output_bytes": output_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--recent-batches", type=int, default=2)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--max-output-bytes", type=int, default=100_000_000)
    parser.add_argument("--minimum-per-window", type=int, default=8)
    parser.add_argument("--selection", choices=("rising", "high_other"),
                        default="rising")
    parser.add_argument("--mode", choices=("extra_updates", "within_batch"),
                        default="extra_updates")
    args = parser.parse_args()
    if any(value < 1 for value in (args.recent_batches, args.repeat_count,
                                    args.max_output_bytes, args.minimum_per_window)):
        parser.error("numeric arguments must be positive")
    paths = sorted(path for path in args.input_dir.glob("*.jsonl")
                   if path.stem.startswith("mi-") or "MI308" in path.stem)
    if not paths:
        parser.error("no MI traces found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [transform_file(path, args.output_dir / path.name,
                              args.recent_batches, args.repeat_count,
                              args.max_output_bytes, args.minimum_per_window,
                              args.selection, args.mode)
               for path in paths]
    print(json.dumps({"selection": args.selection, "mode": args.mode,
                      "hosts": results,
                      "trigger_count": sum(
        len(result["triggers"]) for result in results),
        "extra_updates": sum(result["extra_updates"] for result in results)},
        separators=(",", ":")))


if __name__ == "__main__":
    main()
