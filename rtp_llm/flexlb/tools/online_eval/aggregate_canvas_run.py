#!/usr/bin/env python3
"""Aggregate one online_eval run dir into Sarah-format canvas JSON (stdout).

Run inside a run dir on the remote host:
  cd <run_dir> && python3 aggregate_canvas_run.py
Reads (legacy layout first, consolidated run-root files as fallback):
  load_client/summary.json or client.json
  load_client/slo_batch_analysis.json or client.json's slo_batch_analysis
  load_client/shard_*/per_request.jsonl or per_request.jsonl / per_request.jsonl.gz
  mock_engine.log or mock.json,
  flexlb_logs/flexlb.log* or master.log
Legacy files win whenever they exist: a successful consolidation deletes
them, so a legacy file that is present means fresher data (RUN_DIR reuse).
Outputs meta/summary/batch/per_second/queue_timeseries plus engine_dist
(per-engine routing distribution: requests/Gini/CV/Lorenz/window Gini,
computed from per_request.jsonl ok rows).
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

run_dir = os.getcwd()


def load_json(path):
    """Defensive JSON loader: missing/truncated file -> None (fall back)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_ok(d):
    """Success row predicate (same rule as per_second bucketing below)."""
    err = d.get("error") or ""
    return d.get("status") == "ok" or (
        not err and d.get("status") not in ("schedule_error",)
    )


# ---- inputs: legacy layout first, consolidated run-root fallback ----
legacy_summary = load_json("load_client/summary.json")
if legacy_summary:
    summary = legacy_summary
    client_json = {}
else:
    client_json = load_json("client.json") or {}
    summary = client_json
slo = load_json("load_client/slo_batch_analysis.json")
if not slo and not legacy_summary:
    # Only read the merged copy from client.json when it is the summary source;
    # mixing a fresh legacy summary with a stale client.json slo would leak
    # the previous run's data into this one.
    slo = client_json.get("slo_batch_analysis")
if not slo:
    slo = {}

# ---- per_second from per_request.jsonl (bucket by wall-clock send time) ----
# Legacy shard files first (deleted by consolidation, so their presence means
# fresher data), then the run-root merged file (plain or gzip).
rows = []
per_request_files = sorted(glob.glob("load_client/shard_*/per_request.jsonl"))
if not per_request_files:
    per_request_files = sorted(glob.glob("load_client/per_request.jsonl"))
if not per_request_files:
    if os.path.isfile("per_request.jsonl"):
        per_request_files = ["per_request.jsonl"]
    elif os.path.isfile("per_request.jsonl.gz"):
        per_request_files = ["per_request.jsonl.gz"]
for f in per_request_files:
    opener = gzip.open if f.endswith(".gz") else open
    with opener(f, "rt", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
epoch0 = min((d.get("send_start_epoch_ms", 0) for d in rows), default=0)
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
    if is_ok(d):
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

# ---- queue_timeseries from java_mock_stats (legacy log first, mock.json) ----
mock_stats = []
if os.path.isfile("mock_engine.log"):
    kv_pair_re = re.compile(r"(\w+)=([\d.]+)")
    for line in open("mock_engine.log", errors="replace"):
        if "java_mock_stats" not in line:
            continue
        mock_stats.append(dict(kv_pair_re.findall(line)))
else:
    mock_payload = load_json("mock.json") or {}
    mock_stats = mock_payload.get("stats") or []
queue_ts = []
t0 = None
for kv in mock_stats:
    ts = int(float(kv.get("ts_epoch_ms", 0)))
    if t0 is None:
        t0 = ts
    queue_ts.append(
        {
            "t_offset_s": round((ts - t0) / 1000),
            "prefill_waiting": int(float(kv.get("prefill_waiting", 0))),
            "prefill_running": int(float(kv.get("prefill_running", 0))),
            "prefill_running_reqs": int(float(kv.get("prefill_running_reqs", 0))),
            "max_prefill_waiting": int(float(kv.get("max_prefill_waiting", 0))),
            "decode_waiting": int(float(kv.get("decode_waiting", 0))),
            "decode_running": int(float(kv.get("decode_running", 0))),
            "cum_prefill_batches": int(float(kv.get("prefill_batches", 0))),
            "cum_enqueued_requests": int(float(kv.get("enqueued_requests", 0))),
            "cum_avg_batch_size": float(kv.get("avg_batch_size", 0)),
            "heap_used_mb": int(float(kv.get("heap_used_mb", 0))),
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
# Legacy flexlb_logs/flexlb.log* first; master.log (the consolidated merge)
# carries the same flexlb_batch_dispatch lines as the fallback.
dec_re = re.compile(r"flexlb_batch_dispatch .*?reason=(\w+) batch_size=(\d+)")
hist = Counter()
reason_hist = defaultdict(Counter)
log_files = glob.glob("flexlb_logs/flexlb.log*")
if not log_files and os.path.isfile("master.log"):
    log_files = ["master.log"]
for f in log_files:
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

# ---- engine_dist: per-engine routing distribution (from per_request rows) ----
# Only ok rows count, matching JavaLoadClient's loadBalanceSummary. Mock
# java_mock_stats is cluster-aggregate only, so per-engine utilization and KV
# time series are not computable here (noted, not fabricated).


def gini_coeff(values):
    """Gini coefficient (ascending formula); None when empty/zero-sum."""
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    total = sum(xs)
    if total <= 0:
        return None
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return round((2.0 * weighted) / (n * total) - (n + 1.0) / n, 4)


def cv_coeff(values):
    """Population coefficient of variation; None when empty/zero-mean."""
    if not values:
        return None
    n = len(values)
    mean = sum(values) / float(n)
    if mean == 0:
        return None
    var = sum((x - mean) ** 2 for x in values) / float(n)
    return round((var**0.5) / mean, 3)


def lorenz_pct(values):
    """21-point cumulative share (0..100 step 5), lightest engine first."""
    if not values:
        return []
    xs = sorted(values)
    total = sum(xs)
    if total <= 0:
        return []
    pts = []
    for k in range(21):
        cut = int(round(k * 0.05 * len(xs)))
        pts.append(round(100.0 * sum(xs[:cut]) / total, 2))
    return pts


ed_notes = [
    "prefill_util_pct/decode_util_pct/decode_kv: mock java_mock_stats is "
    "cluster-aggregate only -> omitted"
]
engine_dist = {"notes": ed_notes}
if rows:
    p_count = Counter()
    d_count = Counter()
    p_tokens = defaultdict(float)
    win_p = defaultdict(Counter)
    win_d = defaultdict(Counter)
    for d in rows:
        if not is_ok(d):
            continue
        p = d.get("prefill") or ""
        de = d.get("decode") or ""
        if p:
            p_count[p] += 1
            p_tokens[p] += d.get("input_len", 0) or 0
        if de:
            d_count[de] += 1
        t_ms = d.get("send_start_epoch_ms")
        if t_ms is not None:
            w = int((t_ms - epoch0) // 3000)
            if p:
                win_p[w][p] += 1
            if de:
                win_d[w][de] += 1

    p_vals = sorted(p_count.values(), reverse=True)
    d_vals = sorted(d_count.values(), reverse=True)
    engine_dist["prefill"] = {
        "engine_count": len(p_count),
        "requests_per_engine": p_vals,
        "total": sum(p_vals),
        "gini_cum": gini_coeff(p_vals),
        "cv": cv_coeff(p_vals),
    }
    engine_dist["decode"] = {
        "engine_count": len(d_count),
        "requests_per_engine": d_vals,
        "total": sum(d_vals),
        "gini_cum": gini_coeff(d_vals),
        "cv": cv_coeff(d_vals),
    }
    all_w = sorted(set(win_p) | set(win_d))
    engine_dist["window_gini"] = {
        "t": [str(w * 3) for w in all_w],
        "prefill": [
            gini_coeff(win_p[w].values()) if win_p.get(w) else None for w in all_w
        ],
        "decode": [
            gini_coeff(win_d[w].values()) if win_d.get(w) else None for w in all_w
        ],
    }
    engine_dist["lorenz"] = {
        "x_pct": list(range(0, 101, 5)),
        "prefill_y_pct": lorenz_pct(p_vals),
        "decode_y_pct": lorenz_pct(d_vals),
    }
    if p_tokens:
        engine_dist["prefill_tokens_per_engine"] = [
            round(v, 1) for v in sorted(p_tokens.values(), reverse=True)
        ]
else:
    ed_notes.append("per_request.jsonl not found/empty: engine_dist omitted")

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
    "engine_dist": engine_dist,
}
json.dump(out, sys.stdout)
