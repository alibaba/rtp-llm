#!/usr/bin/env python3
"""Aggregate one online_eval run dir into Sarah-format canvas JSON (stdout).

Run inside a run dir on the remote host:
  cd <run_dir> && python3 aggregate_canvas_run.py
Reads (legacy layout first, consolidated run-root files as fallback):
  load_client/summary.json or client.json
  load_client/slo_batch_analysis.json or client.json's slo_batch_analysis
  load_client/shard_*/per_request.jsonl or per_request.jsonl / per_request.jsonl.gz
  mock_engine.log or mock.json (stats + final_snapshot),
  flexlb_logs/flexlb.log* or master.log (dispatch lines + server-schedule-latency rows),
  master.json (inflight_timeseries G4 / prometheus_timeseries G3),
  run_meta.json (process_usage G5).
Legacy files win whenever they exist: a successful consolidation deletes
them, so a legacy file that is present means fresher data (RUN_DIR reuse).
Outputs meta/summary/batch/per_second (schedule + e2e/ttft percentiles)/
queue_timeseries/engine_dist (requests / tokens / busy-time utilization,
per-engine Gini/CV/Lorenz/window Gini) plus compact time series:
stage_latency_ts (master 10s stage p95 rows), engine_exec_ts (mock
prefill/decode execution windows), process_ts (mock/master/client CPU+RSS),
inflight_ts (G4 scheduler/prefill/decode), inflight_age_ts / kv_ts /
batcher_ts (G3 master prometheus). All series are rebased to the first
per-request send time (negative t = pre-send warmup).
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

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
        "e2e": [],
        "ttft": [],
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
        if d.get("total_ms"):
            b["e2e"].append(d["total_ms"])
        if d.get("ttft_ms"):
            b["ttft"].append(d["ttft_ms"])
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
            "e2e_p50": pct(b["e2e"], 0.5),
            "e2e_p95": pct(b["e2e"], 0.95),
            "ttft_p50": pct(b["ttft"], 0.5),
            "ttft_p95": pct(b["ttft"], 0.95),
        }
    )

# ---- queue_timeseries from java_mock_stats (legacy log first, mock.json) ----
mock_payload = load_json("mock.json") or {}
mock_stats = []
if os.path.isfile("mock_engine.log"):
    kv_pair_re = re.compile(r"(\w+)=([\d.]+)")
    for line in open("mock_engine.log", errors="replace"):
        if "java_mock_stats" not in line:
            continue
        mock_stats.append(dict(kv_pair_re.findall(line)))
else:
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
    "tokens: prefill = input_len sum, decode = output_len sum (engine workload)",
    "busy utilization needs mock final_snapshot busy_ms (mock-engine 4b14e05+)",
]
engine_dist = {"notes": ed_notes}
if rows:
    p_count = Counter()
    d_count = Counter()
    p_tokens = defaultdict(float)
    d_tokens = defaultdict(float)
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
            d_tokens[de] += d.get("output_len", 0) or 0
        t_ms = d.get("send_start_epoch_ms")
        if t_ms is not None:
            w = int((t_ms - epoch0) // 3000)
            if p:
                win_p[w][p] += 1
            if de:
                win_d[w][de] += 1

    p_vals = sorted(p_count.values(), reverse=True)
    d_vals = sorted(d_count.values(), reverse=True)
    p_tok_vals = sorted(p_tokens.values(), reverse=True)
    d_tok_vals = sorted(d_tokens.values(), reverse=True)
    engine_dist["prefill"] = {
        "engine_count": len(p_count),
        "requests_per_engine": p_vals,
        "total": sum(p_vals),
        "gini_cum": gini_coeff(p_vals),
        "cv": cv_coeff(p_vals),
        "tokens_per_engine": [round(v, 1) for v in p_tok_vals],
        "tokens_gini_cum": gini_coeff(p_tok_vals),
        "tokens_cv": cv_coeff(p_tok_vals),
    }
    engine_dist["decode"] = {
        "engine_count": len(d_count),
        "requests_per_engine": d_vals,
        "total": sum(d_vals),
        "gini_cum": gini_coeff(d_vals),
        "cv": cv_coeff(d_vals),
        "tokens_per_engine": [round(v, 1) for v in d_tok_vals],
        "tokens_gini_cum": gini_coeff(d_tok_vals),
        "tokens_cv": cv_coeff(d_tok_vals),
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
        "prefill_tokens_y_pct": lorenz_pct(p_tok_vals),
        "decode_tokens_y_pct": lorenz_pct(d_tok_vals),
    }
    if p_tokens:
        engine_dist["prefill_tokens_per_engine"] = [round(v, 1) for v in p_tok_vals]

    # busy-time utilization: per-engine busy_ms from the mock final_snapshot
    # divided by the effective run window. Elapsed spans the first activity
    # seen by the mock (first stats row with enqueued_requests > 0) through
    # the last stats row, so warmup traffic on both sides of the send window
    # is covered on numerator and denominator alike.
    fs_engines = (mock_payload.get("final_snapshot") or {}).get("engines") or []
    stat_ts = []
    first_active_ts = None
    for kv in mock_stats:
        try:
            ts = int(float(kv.get("ts_epoch_ms", 0) or 0))
        except (TypeError, ValueError):
            continue
        stat_ts.append(ts)
        try:
            if int(float(kv.get("enqueued_requests", 0) or 0)) > 0:
                if first_active_ts is None:
                    first_active_ts = ts
        except (TypeError, ValueError):
            pass
    send_max = max((d.get("send_start_epoch_ms", 0) or 0) for d in rows)
    first_ms = (
        min([x for x in (epoch0, first_active_ts) if x])
        if (epoch0 or first_active_ts)
        else None
    )
    last_ms = max([x for x in (send_max, stat_ts[-1] if stat_ts else 0) if x])
    busy_p, busy_d = [], []
    if first_ms and last_ms and last_ms > first_ms and fs_engines:
        elapsed_s = (last_ms - first_ms) / 1000.0
        for eng in fs_engines:
            if not isinstance(eng, dict):
                continue
            busy = eng.get("busy_ms")
            if not isinstance(busy, (int, float)):
                continue  # old mock build without busy_ms
            role = str(eng.get("role") or "").lower()
            pct_v = round(100.0 * float(busy) / (elapsed_s * 1000.0), 2)
            if role == "prefill":
                busy_p.append(pct_v)
            elif role == "decode":
                busy_d.append(pct_v)
        if busy_p or busy_d:
            busy_p.sort(reverse=True)
            busy_d.sort(reverse=True)
            engine_dist["utilization"] = {
                "elapsed_s": round(elapsed_s, 1),
                "prefill": {
                    "per_engine_pct": busy_p,
                    "gini_cum": gini_coeff(busy_p),
                    "cv": cv_coeff(busy_p),
                },
                "decode": {
                    "per_engine_pct": busy_d,
                    "gini_cum": gini_coeff(busy_d),
                    "cv": cv_coeff(busy_d),
                },
                "note": (
                    "prefill: busy= batch exec ms (maxPrefillConcurrency=1, "
                    "<=100%); decode: busy= request exec ms summed under soft "
                    "concurrency (value = avg concurrent requests, may exceed "
                    "100%)"
                ),
            }
        else:
            ed_notes.append(
                "final_snapshot engines carry no busy_ms (old mock build): "
                "utilization omitted"
            )
else:
    ed_notes.append("per_request.jsonl not found/empty: engine_dist omitted")

# ---- compact time series: G3/G4/G5 + log rows, rebased to epoch0 ----------
# All new series share one time axis: seconds since the first per-request
# send (epoch0). Negative t = pre-send warmup. A series whose source file is
# missing comes out empty; the generator renders charts conditionally.

master_json = load_json("master.json") or {}
run_meta = load_json("run_meta.json") or {}
prom_ts = master_json.get("prometheus_timeseries") or []


def rel_axis(pts):
    """[(epoch_ms, value)] -> [(t_s, value)] on the per-request send axis.

    Falls back to each series' own first sample when per_request rows are
    absent (epoch0 == 0).
    """
    if not pts:
        return []
    anchor = epoch0 or pts[0][0]
    return [(round((ts - anchor) / 1000.0, 1), v) for ts, v in pts]


def prom_ts_extract(base_name, agg="sum"):
    """G3 prometheus timeline -> [(epoch_ms, value)] for one metric.

    Label variants of base_name are folded per sample by agg: "sum" for
    per-engine gauges (queue depth, KV tokens), "avg" for ratios, "max" for
    max-age gauges.
    """
    pts = []
    for grp in prom_ts:
        if not isinstance(grp, dict):
            continue
        metrics = grp.get("metrics")
        if not isinstance(metrics, dict):
            continue
        try:
            ts = float(grp.get("ts", 0) or 0)
        except (TypeError, ValueError):
            continue
        vals = [
            float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and str(k).split("{", 1)[0] == base_name
        ]
        if not vals:
            continue
        if agg == "max":
            v = max(vals)
        elif agg == "avg":
            v = sum(vals) / len(vals)
        else:
            v = sum(vals)
        pts.append((ts, v))
    pts.sort(key=lambda p: p[0])
    return pts


# master 10s ServerScheduleLatencyRecorder rows (SERVER_LAT). The row itself
# carries no ts: parse the log-line datetime prefix (written by the same host
# the aggregation runs on, so local tz matches). Prefix-less rows are stapled
# onto the 10s grid around their anchored neighbours, then the whole set is
# re-sorted by ts (sorted glob order puts the current flexlb.log first).
LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[.,](\d{3})")
SERVER_LAT_LINE_RE = re.compile(
    r"flexlb_server_schedule_latency count=\d+ arrival_qps=[\d.]+ "
    r"completion_qps=[\d.]+ server_p50_ms=([\d.]+) server_p95_ms=([\d.]+) "
    r"server_p99_ms=([\d.]+) grpc_queue_p95_ms=([\d.]+) "
    r"route_submit_p95_ms=([\d.]+) batch_wait_p95_ms=([\d.]+) "
    r"dispatch_ack_p95_ms=([\d.]+) ack_response_p95_ms=([\d.]+)"
)
stage_rows = []
for f in log_files:
    with open(f, errors="replace") as stream:
        for line in stream:
            if "flexlb_server_schedule_latency" not in line:
                continue
            m = SERVER_LAT_LINE_RE.search(line)
            if not m:
                continue
            ts = None
            tm = LOG_TS_RE.match(line)
            if tm:
                try:
                    ts = (
                        datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                        + int(tm.group(2)) / 1000.0
                    ) * 1000.0
                except ValueError:
                    ts = None
            stage_rows.append((ts, [float(m.group(i)) for i in range(1, 9)]))
if stage_rows:
    anchored = [(i, ts) for i, (ts, _) in enumerate(stage_rows) if ts is not None]
    if anchored:
        first_i, first_ts = anchored[0]
        for i in range(first_i):
            stage_rows[i] = (first_ts - (first_i - i) * 10_000.0, stage_rows[i][1])
        prev_i, prev_ts = anchored[0]
        for i, ts in anchored[1:]:
            span = i - prev_i
            step = (ts - prev_ts) / span if span else 10_000.0
            for k in range(prev_i + 1, i):
                stage_rows[k] = (prev_ts + (k - prev_i) * step, stage_rows[k][1])
            prev_i, prev_ts = i, ts
        for i in range(prev_i + 1, len(stage_rows)):
            stage_rows[i] = (prev_ts + (i - prev_i) * 10_000.0, stage_rows[i][1])
        stage_rows.sort(key=lambda r: r[0])
    else:
        stage_rows = [
            ((i + 1) * 10_000.0, fields) for i, (_, fields) in enumerate(stage_rows)
        ]
stage_latency_ts = [
    {
        "t": t,
        "server_p50_ms": round(f[0], 1),
        "server_p95_ms": round(f[1], 1),
        "server_p99_ms": round(f[2], 1),
        "grpc_queue_p95_ms": round(f[3], 1),
        "route_submit_p95_ms": round(f[4], 1),
        "batch_wait_p95_ms": round(f[5], 1),
        "dispatch_ack_p95_ms": round(f[6], 1),
        "ack_response_p95_ms": round(f[7], 1),
    }
    for t, f in rel_axis(stage_rows)
]

# mock engine execution windows (java_mock_stats): decode_exec_* has always
# been there; prefill_exec_* only exists on builds >= 4b14e05 (columns are
# dropped wholesale on old runs instead of zero-filling).
exec_pts = []
any_prefill_exec = False
for kv in mock_stats:
    try:
        ts = int(float(kv.get("ts_epoch_ms", 0) or 0))
    except (TypeError, ValueError):
        continue
    if not ts:
        continue
    if "prefill_exec_p50" in kv:
        any_prefill_exec = True
    exec_pts.append(
        (
            ts,
            (
                int(float(kv.get("decode_exec_p50", 0) or 0)),
                int(float(kv.get("decode_exec_p95", 0) or 0)),
                int(float(kv.get("prefill_exec_p50", 0) or 0)),
                int(float(kv.get("prefill_exec_p95", 0) or 0)),
            ),
        )
    )
engine_exec_ts = []
for t, (d50, d95, p50, p95) in rel_axis(exec_pts):
    row = {"t": t, "decode_exec_p50_ms": d50, "decode_exec_p95_ms": d95}
    if any_prefill_exec:
        row["prefill_exec_p50_ms"] = p50
        row["prefill_exec_p95_ms"] = p95
    engine_exec_ts.append(row)

# G5 process usage (run_meta.json process_usage; legacy raw poller file as
# fallback): client_* shard pollers are averaged into one client series per
# whole second; a role missing at a given second simply omits its keys.
proc_entries = []
for entry in run_meta.get("process_usage") or []:
    if not isinstance(entry, dict):
        continue
    label = str(entry.get("label", ""))
    if label == "mock":
        group = "mock"
    elif label == "master":
        group = "master"
    elif label.startswith("client"):
        group = "client"
    else:
        continue
    try:
        proc_entries.append(
            (
                int(float(entry.get("ts_epoch_ms", 0) or 0)),
                group,
                float(entry.get("cpu_pct", 0) or 0),
                float(entry.get("rss_kb", 0) or 0),
            )
        )
    except (TypeError, ValueError):
        continue
if not proc_entries and os.path.isfile("process_usage_timeseries.txt"):
    kv_re = re.compile(
        r"ts_epoch_ms=(\d+) label=(\S+) pid=\d+ " r"cpu_pct=(-?[\d.]+) rss_kb=(-?\d+)"
    )
    for line in open("process_usage_timeseries.txt", errors="replace"):
        m = kv_re.search(line)
        if not m:
            continue
        label = m.group(2)
        group = (
            "mock"
            if label == "mock"
            else (
                "master"
                if label == "master"
                else "client" if label.startswith("client") else None
            )
        )
        if group:
            proc_entries.append(
                (int(m.group(1)), group, float(m.group(3)), float(m.group(4)))
            )
process_ts = []
if proc_entries:
    anchor = epoch0 or min(e[0] for e in proc_entries)
    by_t = defaultdict(lambda: defaultdict(list))
    for ts, group, cpu, rss in proc_entries:
        by_t[int((ts - anchor) // 1000)][group].append((cpu, rss))
    for t in sorted(by_t):
        row = {"t": t}
        for group in ("mock", "master", "client"):
            samples = by_t[t][group]
            if samples:
                row[group + "_cpu_pct"] = round(
                    sum(s[0] for s in samples) / len(samples), 1
                )
                row[group + "_rss_mb"] = round(
                    sum(s[1] for s in samples) / len(samples) / 1024.0, 1
                )
        if len(row) > 1:
            process_ts.append(row)

# G4 inflight snapshots: scheduler in-flight plus per-endpoint batch/request
# counts summed cluster-wide.
inflight_pts = []
for grp in master_json.get("inflight_timeseries") or []:
    if not isinstance(grp, dict):
        continue
    try:
        ts = int(float(grp.get("ts_epoch_ms", 0) or 0))
    except (TypeError, ValueError):
        continue
    if not ts:
        continue
    infl = grp.get("inflight")
    if not isinstance(infl, dict):
        continue
    try:
        sched = int(infl.get("scheduler_inflight", 0) or 0)
        p_batches = sum(
            int(e.get("inflight_batches", 0) or 0)
            for e in infl.get("prefill_endpoints") or []
            if isinstance(e, dict)
        )
        d_reqs = sum(
            int(e.get("inflight_requests", 0) or 0)
            for e in infl.get("decode_endpoints") or []
            if isinstance(e, dict)
        )
    except (TypeError, ValueError):
        continue
    inflight_pts.append((ts, (sched, p_batches, d_reqs)))
inflight_ts = [
    {"t": t, "scheduler": s, "prefill_batches": pb, "decode_requests": dr}
    for t, (s, pb, dr) in rel_axis(inflight_pts)
]

# master-side queue depth + inflight age from the G3 prometheus timeline
# (needs FLEXLB_MONITOR_MODE=all; per-priority label variants summed).
age_pts = prom_ts_extract("flexlb_app_flexlb_inflight_max_age_ms", agg="max")
inflight_age_ts = [{"t": t, "age_ms": int(round(v))} for t, v in rel_axis(age_pts)]

# KV cache: used / available are per-engine gauges (engineIp labels) summed
# cluster-wide; capacity = used_sum + available_sum. The total gauge is NOT
# per-engine (labels are model+role only, so every engine of a role overwrites
# the same sample) and cannot be summed into a cluster capacity.
kv_used = prom_ts_extract("flexlb_app_cache_used_kv_cache_tokens", agg="sum")
kv_avail = prom_ts_extract("flexlb_app_cache_available_kv_cache_tokens", agg="sum")
kv_ts = []
if kv_used:
    used_by_ts = {ts: v for ts, v in kv_used}
    avail_by_ts = {ts: v for ts, v in kv_avail}
    kv_rows = []
    for ts in sorted(set(used_by_ts) & set(avail_by_ts)):
        used = used_by_ts[ts]
        capacity = used + avail_by_ts[ts]
        if capacity <= 0:
            continue
        kv_rows.append(
            (
                ts,
                {
                    "used_tokens": int(round(used)),
                    "capacity_tokens": int(round(capacity)),
                    "used_pct": round(100.0 * used / capacity, 1),
                },
            )
        )
    kv_ts = [{"t": t, **row} for t, row in rel_axis(kv_rows)]

batcher_pts = prom_ts_extract("flexlb_app_flexlb_batcher_queue_size", agg="sum")
routing_pts = prom_ts_extract("flexlb_app_routing_queue_length", agg="sum")
batcher_ts = []
if batcher_pts or routing_pts:
    b_by_ts = {ts: v for ts, v in batcher_pts}
    r_by_ts = {ts: v for ts, v in routing_pts}
    b_rows = []
    for ts in sorted(set(b_by_ts) | set(r_by_ts)):
        row = {}
        if ts in b_by_ts:
            row["batcher_queue"] = int(round(b_by_ts[ts]))
        if ts in r_by_ts:
            row["routing_queue"] = int(round(r_by_ts[ts]))
        b_rows.append((ts, row))
    batcher_ts = [{"t": t, **row} for t, row in rel_axis(b_rows)]

out = {
    "meta": {
        "run_dir": os.path.basename(run_dir),
        "schedule_only": str(
            (run_meta.get("params") or {}).get("schedule_only", "0")
        ).strip()
        in ("1", "true", "True"),
    },
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
    "stage_latency_ts": stage_latency_ts,
    "engine_exec_ts": engine_exec_ts,
    "process_ts": process_ts,
    "inflight_ts": inflight_ts,
    "inflight_age_ts": inflight_age_ts,
    "kv_ts": kv_ts,
    "batcher_ts": batcher_ts,
}
json.dump(out, sys.stdout)
