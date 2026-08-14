#!/usr/bin/env python3
"""Aggregate one online_eval run dir into Sarah-format canvas JSON (stdout).

Run inside a run dir on the remote host:
  cd <run_dir> && python3 aggregate_canvas_run.py
Reads: load_client/shard_*/per_request.jsonl, mock_engine.log,
       flexlb_logs/flexlb.log*, load_client/summary.json,
       load_client/slo_batch_analysis.json
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

run_dir = os.getcwd()

summary = json.load(open("load_client/summary.json"))
slo = json.load(open("load_client/slo_batch_analysis.json"))

# ---- per_second from per_request.jsonl (bucket by wall-clock send time) ----
rows = []
for f in sorted(glob.glob("load_client/shard_*/per_request.jsonl")):
    for line in open(f):
        rows.append(json.loads(line))
epoch0 = min(d.get("send_start_epoch_ms", 0) for d in rows)
per_sec = defaultdict(
    lambda: {
        "arrivals": 0,
        "success": 0,
        "errors": 0,
        "err_no_decode": 0,
        "err_queue_full": 0,
        "err_deadline": 0,
        "err_preempted": 0,
        "err_yielded": 0,
        "err_other": 0,
        "sched": [],
    }
)
for d in rows:
    t = int((d.get("send_start_epoch_ms", epoch0) - epoch0) // 1000)
    b = per_sec[t]
    b["arrivals"] += 1
    err = d.get("error") or ""
    if d.get("status") == "ok" or (
        not err and d.get("status") not in ("schedule_error",)
    ):
        b["success"] += 1
        b["sched"].append(d.get("schedule_ms", 0))
    else:
        b["errors"] += 1
        # Auto-TPM eviction terminals (checked first so they never fall into
        # err_other): 8429 accepted-eviction ("preempted by higher-priority
        # request N") and yielded_8400 ("yielded to higher-priority request N",
        # carried on the retryable NO_AVAILABLE_WORKER code).
        if "preempted by higher-priority" in err or "8429" in err:
            b["err_preempted"] += 1
        elif "yielded to higher-priority" in err:
            b["err_yielded"] += 1
        elif "NO_DECODE_WORKER" in err or "NO_AVAILABLE_WORKER" in err:
            b["err_no_decode"] += 1
        elif "queue full" in err:
            b["err_queue_full"] += 1
        elif "SLO expired" in err or "deadline" in err:
            b["err_deadline"] += 1
        else:
            b["err_other"] += 1


def pct(v, p):
    if not v:
        return 0
    v = sorted(v)
    return round(v[min(len(v) - 1, int(len(v) * p))], 1)


per_second = []
for t in sorted(per_sec):
    b = per_sec[t]
    per_second.append(
        {
            "t": t,
            "arrivals": b["arrivals"],
            "success": b["success"],
            "errors": b["errors"],
            "err_no_decode": b["err_no_decode"],
            "err_queue_full": b["err_queue_full"],
            "err_deadline": b["err_deadline"],
            "err_preempted": b["err_preempted"],
            "err_yielded": b["err_yielded"],
            "err_other": b["err_other"],
            "sched_p50": pct(b["sched"], 0.5),
            "sched_p95": pct(b["sched"], 0.95),
            "sched_p99": pct(b["sched"], 0.99),
        }
    )

# ---- queue_timeseries from mock_engine.log java_mock_stats lines ----
queue_ts = []
kv_re = re.compile(r"(\w+)=([\d.]+)")
t0 = None
for line in open("mock_engine.log", errors="replace"):
    if "java_mock_stats" not in line:
        continue
    kv = dict(kv_re.findall(line))
    ts = int(kv.get("ts_epoch_ms", 0))
    if t0 is None:
        t0 = ts
    queue_ts.append(
        {
            "t_offset_s": round((ts - t0) / 1000),
            "prefill_waiting": int(kv.get("prefill_waiting", 0)),
            "prefill_running": int(kv.get("prefill_running", 0)),
            "prefill_running_reqs": int(kv.get("prefill_running_reqs", 0)),
            "max_prefill_waiting": int(kv.get("max_prefill_waiting", 0)),
            "decode_waiting": int(kv.get("decode_waiting", 0)),
            "decode_running": int(kv.get("decode_running", 0)),
            "cum_prefill_batches": int(kv.get("prefill_batches", 0)),
            "cum_enqueued_requests": int(kv.get("enqueued_requests", 0)),
            "cum_avg_batch_size": float(kv.get("avg_batch_size", 0)),
            "heap_used_mb": int(kv.get("heap_used_mb", 0)),
        }
    )

# per-interval batch rate / incremental avg batch size from cumulative counters
prev_b, prev_r = 0, 0
for q in queue_ts:
    db = q["cum_prefill_batches"] - prev_b
    dr = q["cum_enqueued_requests"] - prev_r
    q["interval_batches"] = db
    q["interval_avg_batch_size"] = round(dr / db, 2) if db > 0 else 0
    prev_b, prev_r = q["cum_prefill_batches"], q["cum_enqueued_requests"]

# ---- batch size histogram + dispatch reason from flexlb structured logs ----
dec_re = re.compile(r"flexlb_batch_dispatch .*?reason=(\w+) batch_size=(\d+)")
hist = Counter()
reason_hist = defaultdict(Counter)
for f in glob.glob("flexlb_logs/flexlb.log*"):
    for line in open(f, errors="replace"):
        m = dec_re.search(line)
        if m:
            reason, size = m.group(1), int(m.group(2))
            hist[size] += 1
            reason_hist[reason][size] += 1

batch_distribution = {
    "histogram": {str(k): hist[k] for k in sorted(hist)},
    "by_reason": {r: {str(k): c[k] for k in sorted(c)} for r, c in reason_hist.items()},
}

out = {
    "meta": {"run_dir": os.path.basename(run_dir)},
    "summary": {
        "total_requests": summary.get("total_requests"),
        "success_count": summary.get("success_count"),
        "error_count": summary.get("error_count"),
        "error_rate": summary.get("error_rate"),
        "actual_send_qps": summary.get("actual_send_qps"),
        "client_send_peak_qps": summary.get("client_send_peak_qps"),
        "trace_due_peak_qps": summary.get("trace_due_peak_qps"),
        "server_arrival_qps": summary.get("server_arrival_qps"),
        "server_completion_qps": summary.get("server_completion_qps"),
        "schedule_latency_ms": summary.get("schedule_latency_ms"),
        "server_stage_latency_ms": summary.get("server_stage_latency_ms"),
        "validity_checks": summary.get("validity_checks"),
        "test_valid": summary.get("test_valid"),
    },
    "batch": {
        "config": slo.get("config"),
        "decisions": {
            k: v
            for k, v in slo.get("decisions", {}).items()
            if k != "invariant_violation_samples"
        },
        "completions": slo.get("completions"),
        "mock_last": slo.get("mock", {}).get("last"),
        "distribution": batch_distribution,
    },
    "per_second": per_second,
    "queue_timeseries": queue_ts,
}
json.dump(out, sys.stdout)
