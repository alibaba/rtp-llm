#!/usr/bin/env python3
"""Inspect GPU idle intervals in all-rank RTP target Prefill traces.

This diagnostic uses the same CPU launch correlation and matched request
indices as analyze_r41_aligned_phases.py. A gap is the interval between the
union of adjacent target-launched GPU kernels, not necessarily CPU idle time.
"""

import argparse
import gzip
import io
import json
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

from analyze_module_ranges import scope_key
from analyze_phase_by_launch_correlation import family


TARGET_SCOPE = "executor.mtp.prefill_step(target_model_forward)"


def describe(kernel):
    return {
        "family": family(kernel["name"]),
        "module": kernel["module"],
        "symbol": kernel["name"][:160],
    }


def analyze_rank(events, indices):
    phases = sorted(
        (event for event in events if event.get("cat") == "cpu_op"
         and event.get("name") == TARGET_SCOPE),
        key=lambda event: event["ts"],
    )
    annotations = defaultdict(list)
    launches = defaultdict(list)
    for event in events:
        category = event.get("cat")
        if category == "user_annotation" and event.get("name", "").startswith("RTP::"):
            annotations[(event.get("pid"), event.get("tid"))].append(event)
        elif category in {"cuda_runtime", "cuda_driver"}:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation].append(event)

    result = {}
    for index in indices:
        phase = phases[index]
        phase_start = phase["ts"]
        phase_end = phase_start + phase["dur"]
        kernels = []
        ambiguous = 0
        for event in events:
            if event.get("cat") != "kernel":
                continue
            correlation = event.get("args", {}).get("correlation")
            matching = [launch for launch in launches.get(correlation, ())
                        if phase_start <= launch["ts"] < phase_end]
            if not matching:
                continue
            if len(matching) != 1:
                ambiguous += 1
                continue
            launch = matching[0]
            scopes = [annotation for annotation in annotations[(launch.get("pid"), launch.get("tid"))]
                      if annotation["ts"] <= launch["ts"] < annotation["ts"] + annotation["dur"]]
            kernels.append({"ts": event["ts"], "end": event["ts"] + event["dur"],
                            "name": event["name"], "module": scope_key(scopes)})
        if not kernels:
            raise ValueError(f"target request {index} has no correlated GPU kernel")
        kernels.sort(key=lambda kernel: (kernel["ts"], kernel["end"]))
        first = kernels[0]["ts"]
        end = kernels[0]["end"]
        preceding = kernels[0]
        gaps = []
        for kernel in kernels[1:]:
            if kernel["ts"] > end:
                gaps.append({"offset_ms": (end - first) / 1000,
                             "duration_ms": (kernel["ts"] - end) / 1000,
                             "before": describe(preceding), "after": describe(kernel)})
            if kernel["end"] > end:
                end = kernel["end"]
                preceding = kernel
        span_ms = (end - first) / 1000
        gap_ms = sum(gap["duration_ms"] for gap in gaps)
        result[index] = {
            "gpu_span_ms": span_ms,
            "gpu_union_ms": span_ms - gap_ms,
            "gpu_idle_ms": gap_ms,
            "gpu_idle_fraction": gap_ms / span_ms,
            "kernel_count": len(kernels),
            "ambiguous_launch_count": ambiguous,
            "gaps": sorted(gaps, key=lambda gap: gap["duration_ms"], reverse=True),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tar", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    matched = audit["matched_requests"]
    if not matched:
        raise ValueError("no all-rank matched requests in audit")
    ranks = [{} for _ in range(8)]
    with tarfile.open(args.trace_tar, "r:gz") as archive:
        members = [member for member in archive if member.isfile()]
        for rank in range(8):
            found = [member for member in members
                     if re.search(rf"_wr{rank}_\d+\.json$", member.name)]
            if len(found) != 1:
                raise ValueError(f"expected one JSON trace for rank {rank}: {found}")
            data = json.load(io.BytesIO(archive.extractfile(found[0]).read()))
            indices = {row["rank_trace_indices"][rank] for row in matched}
            ranks[rank] = analyze_rank(data["traceEvents"], indices)

    rows = []
    for item in matched:
        per_rank = [ranks[rank][item["rank_trace_indices"][rank]] for rank in range(8)]
        worst = max(range(8), key=lambda rank: per_rank[rank]["gpu_span_ms"])
        expected = item["phases"]["target"]["max_rank_gpu_span_ms"]
        if abs(per_rank[worst]["gpu_span_ms"] - expected) > 0.001:
            raise ValueError(f"GPU span mismatch with phase audit: {per_rank[worst]['gpu_span_ms']} vs {expected}")
        rows.append({"worst_rank": worst, "max_rank_gpu_span_ms": expected,
                     "max_rank_gpu_idle_ms": per_rank[worst]["gpu_idle_ms"],
                     "max_rank_gpu_union_ms": per_rank[worst]["gpu_union_ms"],
                     "per_rank": per_rank})
    output = {
        "method": "target CPU scope -> CUDA launch correlation -> GPU kernel union per rank",
        "caveat": "idle means no correlated target kernel on that GPU; it can include host launch gaps, synchronization, other-stream work, and phase boundaries",
        "trace_tar": str(args.trace_tar), "audit": str(args.audit),
        "matched_requests": rows,
        "median_max_rank_gpu_span_ms": statistics.median(row["max_rank_gpu_span_ms"] for row in rows),
        "median_worst_rank_gpu_idle_ms": statistics.median(row["max_rank_gpu_idle_ms"] for row in rows),
        "median_worst_rank_gpu_union_ms": statistics.median(row["max_rank_gpu_union_ms"] for row in rows),
    }
    payload = json.dumps(output, indent=2) + "\n"
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wt") as stream:
            stream.write(payload)
    else:
        args.output.write_text(payload)
    print(json.dumps({"matched": len(rows), "median_span_ms": output["median_max_rank_gpu_span_ms"],
                      "median_idle_ms": output["median_worst_rank_gpu_idle_ms"],
                      "median_union_ms": output["median_worst_rank_gpu_union_ms"]}, indent=2))


if __name__ == "__main__":
    main()
