#!/usr/bin/env python3
"""Reconstruct summary.json from per_request.jsonl for sweeps where load client was killed."""
import json
import statistics
import sys
from pathlib import Path


def reconstruct(per_request_file: str, trace_file: str = "") -> dict:
    records = []
    with open(per_request_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    ok = [r for r in records if r["status"] == "ok"]
    scheduled = [r for r in records if r["status"] in ("ok", "scheduled")]
    ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    total = [r["total_ms"] for r in ok if r["total_ms"] > 0]
    schedule = [r["schedule_ms"] for r in records if r["schedule_ms"] > 0]

    # Compute elapsed time from wall_clock_ts
    if records:
        wtses = [
            r.get("wall_clock_ts", 0) for r in records if r.get("wall_clock_ts", 0) > 0
        ]
        if wtses:
            elapsed = max(wtses) - min(
                r.get("wall_clock_ts", 0) - r.get("total_ms", 0) / 1000.0
                for r in records
                if r.get("wall_clock_ts", 0) > 0
            )
        else:
            elapsed = 0
    else:
        elapsed = 0

    def pct(vals, p):
        if not vals:
            return 0
        sorted_vals = sorted(vals)
        idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
        return round(sorted_vals[idx], 3)

    def lat_summary(vals):
        if not vals:
            return {
                "count": 0,
                "p50": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0,
                "max": 0,
                "mean": 0,
            }
        return {
            "count": len(vals),
            "p50": pct(vals, 50),
            "p90": pct(vals, 90),
            "p95": pct(vals, 95),
            "p99": pct(vals, 99),
            "max": round(max(vals), 3),
            "mean": round(statistics.mean(vals), 3),
        }

    # Count by status
    status_counts = {}
    for r in records:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    # Balance
    prefill_counts = {}
    decode_counts = {}
    for r in ok:
        p = r.get("prefill", "")
        d = r.get("decode", "")
        if p:
            prefill_counts[p] = prefill_counts.get(p, 0) + 1
        if d:
            decode_counts[d] = decode_counts.get(d, 0) + 1

    def balance_summary(counts):
        vals = list(counts.values())
        if not vals:
            return {"counts": {}, "stddev": 0, "max_over_avg": 0}
        avg = statistics.mean(vals)
        return {
            "counts": counts,
            "stddev": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0,
            "max_over_avg": round(max(vals) / avg, 3) if avg > 0 else 0,
        }

    return {
        "trace": trace_file,
        "elapsed_s": round(elapsed, 3),
        "total_requests": len(records),
        "scheduled": len(scheduled),
        "completed": len(ok),
        "errors": len(records) - len(scheduled),
        "offered_qps": round(len(records) / elapsed, 3) if elapsed > 0 else 0,
        "completed_qps": round(len(ok) / elapsed, 3) if elapsed > 0 else 0,
        "sla_ttft_ms": 500.0,
        "sla_violations": sum(1 for r in ok if r["ttft_ms"] > 500),
        "sla_violation_rate": (
            round(sum(1 for r in ok if r["ttft_ms"] > 500) / len(ok), 6) if ok else 0
        ),
        "schedule_latency_ms": lat_summary(schedule),
        "ttft_ms": lat_summary(ttft),
        "total_ms": lat_summary(total),
        "prefill_balance": balance_summary(prefill_counts),
        "decode_balance": balance_summary(decode_counts),
        "status_counts": status_counts,
        "route_path_counts": {"master": len(records)},
    }


def main():
    exp_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent / "run" / "inflight_experiment"
    )

    for speed in [3, 5, 7]:
        sweep_dir = exp_dir / f"sweep_{speed}x"
        per_req = sweep_dir / "load_client" / "per_request.jsonl"
        summary = sweep_dir / "load_client" / "summary.json"

        if not per_req.exists():
            print(f"  {speed}x: no per_request file, skipping")
            continue

        if summary.exists():
            existing = json.loads(summary.read_text())
            if existing.get("total_requests", 0) > 0:
                print(
                    f"  {speed}x: summary already complete ({existing['total_requests']} requests)"
                )
                continue

        s = reconstruct(str(per_req))
        summary.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(
            f"  {speed}x: reconstructed summary - offered={s['offered_qps']}, completed={s['completed_qps']}, ok={s['completed']}/{s['total_requests']}, errors={s['errors']}, elapsed={s['elapsed_s']}s"
        )


if __name__ == "__main__":
    main()
