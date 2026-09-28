#!/usr/bin/env python3
"""Match full-rank MTP Prefill requests by target CPU timestamp."""

import argparse
import gzip
import importlib.util
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927")
DEFAULT_TRACE = ROOT / "timeline-64k-integrated-mtp-local-r41-111112-20260928/traces"

parser = argparse.ArgumentParser()
parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE)
parser.add_argument("--trace-template", default="k3_64k_prefill_r41_wr{rank}_1.json")
parser.add_argument("--output", type=Path)
parser.add_argument("--min-common", type=int, default=2)
parser.add_argument("--match-window-us", type=int, default=2000)
args = parser.parse_args()
if args.min_common < 1 or args.match_window_us <= 0:
    parser.error("invalid matching threshold")

spec = importlib.util.spec_from_file_location("phase", HERE / "analyze_phase_by_launch_correlation.py")
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)

rank_data = []
for rank in range(8):
    path = args.trace_dir / args.trace_template.format(rank=rank)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")) as stream:
        trace = json.load(stream)["traceEvents"]
    target = sorted((e["ts"] for e in trace if e.get("cat") == "cpu_op" and
                     e["name"] == "executor.mtp.prefill_step(target_model_forward)"))
    if not target:
        raise RuntimeError(f"No target Prefill scope in {path}")
    analyzed = phase.analyze_one(path, len(target))
    rank_data.append((target, analyzed["requests"]))

matched = []
used_indices = [set() for _ in rank_data]
for ts in rank_data[0][0]:
    indices = []
    for rank, (timestamps, _) in enumerate(rank_data):
        near = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - ts))
        if abs(timestamps[near] - ts) > args.match_window_us or near in used_indices[rank]:
            break
        indices.append(near)
    if len(indices) != 8:
        continue
    for rank, index in enumerate(indices):
        used_indices[rank].add(index)
    record = {"rank_trace_indices": indices, "target_timestamp_us": ts, "phases": {}}
    for name in ("prefill", "target", "draft"):
        per_rank = [rank_data[rank][1][indices[rank]][name] for rank in range(8)]
        worst = max(range(8), key=lambda rank: per_rank[rank]["gpu_span_ms"])
        record["phases"][name] = {
            "worst_rank": worst,
            "max_rank_gpu_span_ms": per_rank[worst]["gpu_span_ms"],
            "max_rank_gpu_union_ms": max(row["gpu_union_ms"] for row in per_rank),
            "worst_rank_family_cumulative_ms": {key: val["cumulative_ms"]
                for key, val in per_rank[worst]["families"].items()},
        }
    matched.append(record)

if len(matched) < args.min_common:
    raise RuntimeError(f"Only {len(matched)} common full-rank requests")
summary = {name: statistics.median(row["phases"][name]["max_rank_gpu_span_ms"]
                                   for row in matched) for name in ("prefill", "target", "draft")}
result = {"input_tokens": 65536, "model_layers": 4, "world_size": 8,
          "kernel_family_classifier": "v2: symbol-level only; fused AttnRes FP8 producer has its own family, generic GEMMs remain unattributed",
          "family_timing_caveat": "cumulative kernel durations can overlap across streams and are not wall time",
          "same_request_match": f"nearest target CPU scope timestamp within {args.match_window_us}us",
          "trace_dir": str(args.trace_dir), "trace_template": args.trace_template,
          "matched_requests": matched, "median_max_rank_gpu_span_ms": summary}
output = args.output or args.trace_dir.parent / "aligned-phase-audit.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"matched": len(matched), "median_max_rank_gpu_span_ms": summary}, indent=2))
