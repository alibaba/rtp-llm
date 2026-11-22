"""Summarize completed local Worker callbacks on the monotonic arrival clock."""

import argparse
import csv
import gzip
import struct
from array import array
from collections import defaultdict
from pathlib import Path

RECORD = struct.Struct(">6q")
SECOND = 1_000_000_000


def quantile(sorted_values, fraction):
    if not sorted_values:
        return 0
    return sorted_values[int((len(sorted_values) - 1) * fraction)] / 1e6


def summarize(source, start, end, *, emit_intervals=False):
    if not (0 <= start < end):
        raise ValueError("specify a nonempty window in seconds since replay start")
    prefix = source.with_name(f"{source.stem}-{start}-{end}")
    seconds = defaultdict(lambda: [0] * (end - start))
    grouped = defaultdict(lambda: [array("q") for _ in range(4)])
    changed = defaultdict(int)
    workers = set()
    intervals = (
        gzip.open(str(prefix) + "-intervals.csv.gz", "wt", compresslevel=1)
        if emit_intervals
        else None
    )
    try:
        if intervals:
            intervals.write("lane,completion_ns,gap_ns,version,membership_delta\n")
        with source.open("rb") as f, Path(str(prefix) + "-workers.csv").open(
            "w"
        ) as metrics:
            writer = csv.writer(metrics)
            writer.writerow(
                [
                    "role",
                    "event",
                    "worker",
                    "success",
                    "success_per_s",
                    "nonempty_delta",
                    "gap_p50_ms",
                    "gap_p95_ms",
                    "gap_p99_ms",
                    "gap_max_ms",
                ]
            )
            (count,) = struct.unpack(">I", f.read(4))
            for _ in range(count):
                name = f.read(struct.unpack(">H", f.read(2))[0]).decode("ascii")
                role, event, worker = name.split("-")
                if name in workers:
                    raise ValueError("duplicate lane " + name)
                workers.add(name)
                key = (role, event)
                prev = last_version = -1
                gaps = []
                n = deltas = 0
                (length,) = struct.unpack(">I", f.read(4))
                for sent, captured, apply, done, version, delta in RECORD.iter_unpack(
                    f.read(length * RECORD.size)
                ):
                    if (
                        not prev < sent <= captured <= apply <= done
                        or apply - captured < 10_000_000
                    ):
                        raise ValueError(
                            "request/reply transport or order violated in " + name
                        )
                    if version < last_version or (
                        version == last_version and event == "status"
                    ):
                        raise ValueError("response version order violated in " + name)
                    last_version = version
                    sec = done // SECOND
                    if start <= sec < end:
                        n += 1
                        deltas += delta
                        seconds[key][sec - start] += 1
                        if prev >= 0:
                            gaps.append(done - prev)
                            if intervals:
                                intervals.write(
                                    f"{name},{done},{done-prev},{version},{delta}\n"
                                )
                        for values, elapsed in zip(
                            grouped[key][1:],
                            [captured - sent, apply - captured, done - apply],
                        ):
                            values.append(elapsed)
                    prev = done
                if not n:
                    raise ValueError(f"no successful callbacks for {name} in window")
                changed[key] += deltas
                grouped[key][0].extend(gaps)
                gaps.sort()
                writer.writerow(
                    [
                        role,
                        event,
                        worker,
                        n,
                        n / (end - start),
                        deltas,
                        *[quantile(gaps, q) for q in (0.5, 0.95, 0.99, 1)],
                    ]
                )
            if f.read(1):
                raise ValueError("unexpected trailing data")
    finally:
        if intervals:
            intervals.close()
    with Path(str(prefix) + "-seconds.csv").open("w") as f:
        writer = csv.writer(f)
        keys = sorted(grouped)
        writer.writerow(["second", *["-".join(k) for k in keys]])
        for sec in range(start, end):
            writer.writerow([sec, *[seconds[k][sec - start] for k in keys]])
    for key in sorted(grouped):
        gap, capture, transit, apply = map(sorted, grouped[key])
        rates = seconds[key]
        print(
            source.stem,
            *key,
            f"window={start}..{end}",
            f"success/s={sum(rates)/len(rates):.2f}",
            f"one_second_range={min(rates)}..{max(rates)}",
            f"gap_ms_p50/p95/p99/max={','.join(f'{quantile(gap,q):.2f}' for q in (.5,.95,.99,1))}",
            f"capture/transit/apply_p95_ms={','.join(f'{quantile(v,.95):.2f}' for v in (capture,transit,apply))}",
            f"nonempty_delta/s={changed[key]/len(rates):.2f}",
        )
    for role, count in (("PREFILL", 30), ("DECODE", 50)):
        for event in ("cache", "status"):
            expected = {f"{role}-{event}-{i}" for i in range(count)}
            actual = {lane for lane in workers if lane.startswith(f"{role}-{event}-")}
            if actual != expected:
                raise ValueError(f"incomplete {role} {event} population")
    return workers


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, nargs="+")
    p.add_argument("--start", type=int, default=240)
    p.add_argument("--end", type=int, default=360)
    p.add_argument(
        "--intervals",
        action="store_true",
        help="also expand every adjacent callback interval as CSV",
    )
    args = p.parse_args()
    for path in args.source:
        summarize(path, args.start, args.end, emit_intervals=args.intervals)
