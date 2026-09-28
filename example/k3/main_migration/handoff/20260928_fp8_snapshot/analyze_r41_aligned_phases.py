#!/usr/bin/env python3
"""Match full-rank MTP Prefill requests by target CPU timestamp."""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path("/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927")
TRACE = ROOT / "timeline-64k-integrated-mtp-local-r41-111112-20260928/traces"
OUT = TRACE.parent / "aligned-phase-audit.json"

spec = importlib.util.spec_from_file_location("phase", ROOT / "analyze_phase_by_launch_correlation.py")
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)

rank_data = []
for rank in range(8):
    path = TRACE / f"k3_64k_prefill_r41_wr{rank}_1.json"
    trace = json.loads(path.read_text())["traceEvents"]
    target = sorted((e["ts"] for e in trace if e.get("cat") == "cpu_op" and
                     e["name"] == "executor.mtp.prefill_step(target_model_forward)"))
    analyzed = phase.analyze_one(path, len(target))
    rank_data.append((target, analyzed["requests"]))

matched = []
for ts in rank_data[0][0]:
    indices = []
    for timestamps, _ in rank_data:
        near = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - ts))
        if abs(timestamps[near] - ts) > 2000:
            break
        indices.append(near)
    if len(indices) != 8:
        continue
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

if len(matched) < 2:
    raise RuntimeError(f"Only {len(matched)} common full-rank requests")
summary = {name: statistics.median(row["phases"][name]["max_rank_gpu_span_ms"]
                                   for row in matched) for name in ("prefill", "target", "draft")}
result = {"input_tokens": 65536, "model_layers": 4, "world_size": 8,
          "same_request_match": "nearest target CPU scope timestamp within 2ms",
          "matched_requests": matched, "median_max_rank_gpu_span_ms": summary}
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"matched": len(matched), "median_max_rank_gpu_span_ms": summary}, indent=2))
