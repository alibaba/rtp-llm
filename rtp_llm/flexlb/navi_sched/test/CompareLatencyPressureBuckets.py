#!/usr/bin/env python3
"""Offline replay of pressure-matched history on top of latency-bucket sampling.

The logged feature count is an admission-time proxy, not fit-time inflight.
"""

import argparse
import collections
import gzip
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


BUCKET_CAPACITY = 64
FIT_INFLIGHT_FIELDS = (
    "fit_inflight_before_valid", "fit_inflight_before_requests",
    "fit_inflight_before_revision", "fit_inflight_before_time_us",
    "fit_inflight_after_valid", "fit_inflight_after_requests",
    "fit_inflight_after_revision", "fit_inflight_after_time_us",
)


def pressure(sample):
    features = sample["features"]
    return features[1] + features[6]


def bucket(count):
    return max(0, count).bit_length() - 1 if count > 0 else -1


def augment(events, extra, selector, replace=False, shared_fit_window=False):
    history = collections.defaultdict(lambda: collections.deque(maxlen=BUCKET_CAPACITY))
    samples = {}
    for event in events:
        if event["type"] == "sample":
            if "history_sequence" in event:
                sequence = event["history_sequence"]
                samples[sequence] = event
                history[bucket(pressure(event))].append(sequence)
            yield event
            continue
        if event["type"] != "update" or "fit_batches" not in event:
            yield event
            continue
        if shared_fit_window and not event.get("fit_inflight_before_valid"):
            yield event
            continue
        if selector == "fit_inflight" and not event.get("fit_inflight_before_valid"):
            yield event
            continue
        latest = samples[event["fit_batches"][-1][-1]]
        batches = []
        for original in event["fit_batches"]:
            current = original[-1]
            if selector == "fit_inflight":
                current_count = event["fit_inflight_before_requests"]
            else:
                reference = samples[current] if selector == "current_sample" else latest
                current_count = pressure(reference)
            current_bucket = bucket(current_count)
            choices = [sequence for sequence in reversed(history[current_bucket])
                       if sequence < current]
            coverage = original[:-1]
            if replace:
                coverage = []
                for index, entries in sorted(history.items()):
                    if index == current_bucket:
                        continue
                    eligible = [sequence for sequence in entries if sequence < current]
                    if eligible:
                        coverage.append(eligible[(current * 1315423911 + index) % len(eligible)])
            # Keep the current sample last: the online fitter weights only that position.
            batches.append(coverage + choices[:extra] + [current])
        yield {**event, "fit_batches": batches}


def load_events(directory, excluded_hosts=()):
    result = {}
    for path in sorted(directory.glob("*.jsonl")):
        if path.stem in excluded_hosts:
            continue
        with path.open(encoding="utf-8") as stream:
            result[path.name] = [json.loads(line) for line in stream]
    if not result:
        raise ValueError("no replay JSONL files")
    return result


def parse_fit_inflight_line(line):
    if "[navi-latency-training-update]" not in line:
        return None
    sequence = re.search(r"\bsequence=(\d+)", line)
    values = {key: int(value) for key, value in re.findall(
        r"\b(fit_inflight_(?:before|after)_(?:valid|requests|revision|time_us))=(-?\d+)", line)}
    if not sequence or any(key not in values for key in FIT_INFLIGHT_FIELDS):
        return None
    return int(sequence.group(1)), values


def attach_fit_inflight(files, log_dir, start_us, end_us):
    coverage = collections.Counter()
    for filename, events in files.items():
        path = log_dir / f"{Path(filename).stem}.log.gz"
        found = {}
        with gzip.open(path, "rt", errors="replace") as stream:
            for line in stream:
                parsed = parse_fit_inflight_line(line)
                if parsed:
                    sequence, values = parsed
                    if sequence in found and found[sequence] != values:
                        raise ValueError(f"conflicting fit inflight for {filename} update {sequence}")
                    found[sequence] = values
        sample_times = {event["history_sequence"]: event["event_time_us"]
                        for event in events if event["type"] == "sample"
                        and "history_sequence" in event}
        for event in events:
            if event["type"] != "update":
                continue
            fit_batches = event.get("fit_batches", [])
            if not fit_batches:
                continue
            sample_time = sample_times[fit_batches[-1][-1]]
            values = found.get(event["sequence"])
            if not values or not values["fit_inflight_before_valid"]:
                if start_us <= sample_time < end_us:
                    raise ValueError(f"missing live fit inflight for {filename} update {event['sequence']}")
                continue
            fit_time = values["fit_inflight_before_time_us"]
            if fit_time <= 0 or values["fit_inflight_after_time_us"] < fit_time:
                raise ValueError(f"invalid fit inflight time for {filename} update {event['sequence']}")
            if not start_us <= fit_time < end_us:
                continue
            event.update(values)
            event["fit_inflight_stable"] = bool(
                values["fit_inflight_after_valid"] and
                values["fit_inflight_before_revision"] ==
                values["fit_inflight_after_revision"] and
                values["fit_inflight_before_requests"] ==
                values["fit_inflight_after_requests"])
            coverage["stable" if event["fit_inflight_stable"] else "unstable"] += 1
    return coverage


def run_replay(binary, directory, output, weight, scale, residual, mode="fixed"):
    subprocess.run([str(binary), str(directory), str(output), str(weight), "20261008",
                    str(scale), str(residual), "all", "0", mode], check=True)


def predictions(path, start_us, end_us, time_basis="event", excluded_hosts=()):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "P" or fields[1] in excluded_hosts:
                continue
            target = float(fields[6])
            event_time = int(fields[5])
            if time_basis == "ready":
                event_time -= round(target * 1000)
            if start_us <= event_time < end_us:
                result[(fields[1], int(fields[4]))] = (target, float(fields[9]),
                                                       float(fields[12]), event_time)
    return result


def summarize(rows):
    if not rows:
        return None
    errors = [predicted - target for target, predicted in rows]
    ordered = sorted(map(abs, errors))
    over = sorted(error for error in errors if error > 0)
    under = sorted(-error for error in errors if error < 0)
    return {"count": len(rows), "mae_ms": sum(ordered) / len(rows),
            "wape": sum(ordered) / sum(target for target, _ in rows),
            "bias_ms": sum(errors) / len(rows),
            "abs_p95_ms": ordered[math.ceil(0.95 * len(rows)) - 1],
            "over_p95_ms": over[math.ceil(0.95 * len(over)) - 1] if over else None,
            "under_p95_ms": under[math.ceil(0.95 * len(under)) - 1] if under else None}


def compare(baseline, candidate, metadata, start_us, host_arch=None):
    if baseline.keys() != candidate.keys():
        raise ValueError("candidate changed the evaluated sample set")
    groups = collections.defaultdict(list)
    for key, (target, predicted, _, event_time) in candidate.items():
        original_target = baseline[key][0]
        if abs(target - original_target) > 0.001:
            raise ValueError("candidate changed a target")
        arch = host_arch[key[0]] if host_arch else (
            "MI" if key[0].startswith(("mi-", "11_")) else "RTX")
        groups[(arch, "all")].append((target, predicted))
        groups[(arch, f"host_{key[0]}")].append((target, predicted))
        groups[(arch, f"window_{(event_time - start_us) // 900_000_000}")].append(
            (target, predicted))
        count, low, high = metadata[key]
        band = "low" if count <= low else "high" if count >= high else "middle"
        groups[(arch, band)].append((target, predicted))
    return {f"{arch}/{segment}": summarize(rows)
            for (arch, segment), rows in sorted(groups.items())}


def make_metadata(files, start_us, end_us, time_basis="event"):
    result = {}
    for filename, events in files.items():
        host = Path(filename).stem
        def sample_time(event):
            if time_basis == "ready":
                return event["event_time_us"] - round(event["target_ms"] * 1000)
            return event["event_time_us"]

        window = [pressure(event) for event in events if event["type"] == "sample"
                  and start_us <= sample_time(event) < end_us]
        if not window:
            continue
        ordered = sorted(window)
        low, high = ordered[len(ordered) // 4], ordered[3 * len(ordered) // 4]
        for event in events:
            if event["type"] == "sample" and start_us <= sample_time(event) < end_us:
                result[(host, event["sequence"])] = (pressure(event), low, high)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--end-us", type=int, required=True)
    parser.add_argument("--extra", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--schemes", nargs="+", choices=("augment", "replace"),
                        default=["augment", "replace"])
    parser.add_argument("--selectors", nargs="+",
                        choices=("current_sample", "update_latest", "fit_inflight"),
                        default=["current_sample", "update_latest"])
    parser.add_argument("--fit-inflight-logs", type=Path)
    parser.add_argument("--weight", type=float, default=4.0)
    parser.add_argument("--scale", type=float, default=16.0)
    parser.add_argument("--residual", type=float, default=0.8)
    parser.add_argument("--mode", default="fixed")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--exclude-hosts", nargs="*", default=[])
    parser.add_argument("--time-basis", choices=("event", "ready"), default="event")
    parser.add_argument("--shared-fit-window", action="store_true")
    parser.add_argument("--allow-counterfactual-baseline", action="store_true")
    args = parser.parse_args()
    if args.end_us <= args.start_us or any(value < 1 for value in args.extra):
        parser.error("invalid window or extra sample count")
    if args.mode not in ("fixed", "production") and not args.mode.startswith(
            "production_cap:"):
        parser.error("mode must be fixed, production, or production_cap:BASE,RESIDUAL")
    if "fit_inflight" in args.selectors and not args.fit_inflight_logs:
        parser.error("fit_inflight selector requires --fit-inflight-logs")
    if args.shared_fit_window and not args.fit_inflight_logs:
        parser.error("shared fit window requires --fit-inflight-logs")
    files = load_events(args.events, set(args.exclude_hosts))
    host_arch = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        host_arch = {}
        for node in manifest["nodes"]:
            deployment = node["deployment"].upper()
            host = node["ip"].replace(".", "_")
            if "MI308" in deployment:
                host_arch[host] = "MI"
            elif "RTX_PRO_5000" in deployment:
                host_arch[host] = "RTX"
        if not all(Path(name).stem in host_arch for name in files):
            raise ValueError("manifest lacks architecture for a replay host")
    fit_inflight_coverage = None
    if args.fit_inflight_logs:
        fit_inflight_coverage = attach_fit_inflight(
            files, args.fit_inflight_logs, args.start_us, args.end_us)
        if "fit_inflight" in args.selectors and not sum(fit_inflight_coverage.values()):
            raise ValueError("no valid live fit inflight updates in selected window")
    metadata = make_metadata(files, args.start_us, args.end_us, args.time_basis)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline_file = root / "baseline.tsv"
        run_replay(args.binary, args.events, baseline_file, args.weight, args.scale,
                   args.residual, args.mode)
        baseline = predictions(baseline_file, args.start_us, args.end_us,
                               args.time_basis, set(args.exclude_hosts))
        if not baseline:
            raise ValueError("no predictions in selected window")
        differences = collections.defaultdict(list)
        for (host, _), (_, predicted, logged, _) in baseline.items():
            differences[host].append(abs(predicted - logged))
        maximum_log_difference = max(max(values) for values in differences.values())
        maximum_host_mean_difference = max(sum(values) / len(values)
                                           for values in differences.values())
        if not args.allow_counterfactual_baseline and (
                maximum_host_mean_difference > 0.1 or maximum_log_difference > 1.0):
            raise ValueError("baseline does not match online log: "
                             f"host mean {maximum_host_mean_difference}, "
                             f"maximum {maximum_log_difference} ms")
        print(json.dumps({"mode": "baseline", "replay_mode": args.mode,
                          "max_log_difference_ms": maximum_log_difference,
                          "max_host_mean_log_difference_ms": maximum_host_mean_difference,
                          "fit_inflight_coverage": fit_inflight_coverage,
                          "metrics": compare(baseline, baseline, metadata,
                                             args.start_us, host_arch)}))
        for scheme in args.schemes:
            for selector in args.selectors:
                for extra in args.extra:
                    candidate_dir = root / f"{scheme}_{selector}_{extra}"
                    candidate_dir.mkdir()
                    for filename, events in files.items():
                        with (candidate_dir / filename).open("w", encoding="utf-8") as stream:
                            for event in augment(events, extra, selector,
                                                 scheme == "replace",
                                                 args.shared_fit_window):
                                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
                    output = root / f"{scheme}_{selector}_{extra}.tsv"
                    run_replay(args.binary, candidate_dir, output, args.weight, args.scale,
                               args.residual, args.mode)
                    candidate = predictions(output, args.start_us, args.end_us,
                                            args.time_basis, set(args.exclude_hosts))
                    print(json.dumps({"scheme": scheme, "mode": selector,
                                      "replay_mode": args.mode, "extra": extra,
                                      "metrics": compare(baseline, candidate, metadata,
                                                         args.start_us, host_arch)}), flush=True)


if __name__ == "__main__":
    main()
