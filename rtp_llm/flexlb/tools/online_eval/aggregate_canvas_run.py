#!/usr/bin/env python3
"""Aggregate one online_eval run dir into Sarah-format canvas JSON (stdout).

Run inside a run dir on the remote host:
  cd <run_dir> && python3 aggregate_canvas_run.py
Reads (consolidated run-root layout, the only supported form):
  client.json (summary source: sh-merged scalar keys + embedded
  server_latency / slo_batch_analysis)
  client_events.jsonl / client_events.jsonl.gz (run root; renamed from
  per_request.jsonl together with the multi-component JSONL event streams)
  engine_events.jsonl / engine_events.jsonl.gz (run root; the engine-side
  per-rid event rows that replaced the mock_prefill_done / mock_decode_done
  stdout trace lines)
  master_events.jsonl — written by THIS script (offline parser) from
  master.log / flexlb_logs (window rows + generation created/retired
  events); master production code is untouched
  load_client/server_latency.json or client.json's server_latency
  mock_engine.log or mock.json (stats + final_snapshot; comparison source
  alongside the jsonl streams — the per-rid data itself is jsonl-only),
  flexlb_logs/flexlb.log* or master.log (dispatch lines + server-schedule-latency rows),
  master.json (inflight_timeseries G4 / prometheus_timeseries G3 /
  counters_timeseries master per-second arrival/completion rates;
  comparison source),
  run_meta.json (process_usage G5).
client_events rows are the sole per-request metrics source; the engine-side
event rows are rid-joined onto them (the same join full_e2e always used).
Both jsonl streams are fail-closed: a missing file or a malformed row is a
hard error, never a silent empty column (no-backward-compat: legacy runs
that predate the jsonl streams are no longer supported).
Outputs meta/summary/batch/per_second (schedule + e2e/ttft percentiles)/
master_arrivals_ts (master-side per-second arrival/completion rates, the
send-series source of record) /queue_timeseries/engine_dist (requests / tokens / busy-time utilization,
per-engine Gini/CV/Lorenz/window Gini) plus compact time series:
stage_latency_ts (master 10s stage p95 rows), engine_exec_ts (mock
prefill/decode execution windows), per_second prefill_exec_* /
decode_exec_* (engine exec percentiles joined by request_id onto the
request-BIRTH second — same axis as e2e/full_e2e, unlike the
completion-window engine_exec_ts), process_ts (mock/master/client CPU+RSS),
inflight_ts (G4 scheduler/prefill batches+requests/decode reserved+
confirmed_running), inflight_age_ts / kv_ts /
batcher_ts / dispatch_reason_ts (G3 master prometheus; the reason series is
per-second dispatch rate derived from the dispatch_reason_total counters).
cancel_qps_ts additionally carries master/prefill/decode cancel split rates
(census unknown/finished/tombstone diff for the master side; per-engine
cancelled_rids matched against master terminal lines for prefill/decode).
All series are rebased to the first
per-request send time (negative t = pre-send warmup).

Phase A (aggregator-side unification): the summary section additionally
carries validity_checks / test_valid (seven checks incl. the
client-vs-mock token reconciliation, now also produced for
single-worker runs), quick-stats (actual_send_qps, success/error/
completed qps, elapsed_s, counts, error_rate) and full-run percentiles for
ttft/e2e/schedule (schedule dual-source adjudication via
schedule_latency_source) — all computed from the merged client_events rows
plus the server_latency terminal state; rows missing -> None (no
summary.json passthrough, no-backward-compat).
"""
import glob
import gzip
import json
import os
import re
import sys
from bisect import bisect_right
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


# ---- shared-impl-begin: consolidate_run_outputs.py 经 exec 切块复用本段 ----
# （is_ok / 17 桶 classify_error / Phase A 统计原语；修改本段时同步检查
# consolidate_run_outputs.py 的 load_shared_impl 切块边界：本段只能依赖
# re / Counter 与内置函数，不得引用本文件其它名字）。


def is_ok(d):
    """Success row predicate (same rule as per_second bucketing below)."""
    err = d.get("error") or ""
    return d.get("status") == "ok" or (
        not err and d.get("status") not in ("schedule_error",)
    )


# ---- err_other 细分：具名子桶匹配规则（先具名、后残渣） ----
# 真实错误文本依据（20260827 本地 per_request.jsonl 实测 + flexlb-sync
# StrategyErrorType/AdmissionFailure/RequestLifecycleCoordinator 源码）：
#   * gRPC 传输层（load client e.toString()）：
#       "io.grpc.StatusRuntimeException: UNAVAILABLE: io exception"（实测大头）、
#       "...: UNAVAILABLE: Network closed for unknown reason"、
#       "...: INTERNAL: RST_STREAM closed by remote peer"、GOAWAY/CANCELLED 同族
#   * master schedule_error（StrategyErrorType.buildErrorMessage）：enum 名
#     （全大写，如 "RESOURCE_EXHAUSTED"）、detail 文本（如 "admission
#     capacity is temporarily exhausted; trigger=..."）或 JSON
#     status_name（{"status_name":"GatewayTimeout",...}）；旧的小写子串链
#     漏掉全部 enum 名形态 —— 这是 err_other 巨大的主因
#   * load client 自身："stream completed with zero outputs"（empty_response）
# 具名子桶清单与优先级（前者先匹配）：
#   err_backpressure    master 准入/容量拒绝族 8430/8431/8432
#                       （PRIORITY_ADMISSION_REJECTED / RESOURCE_EXHAUSTED /
#                       ADMISSION_UNAVAILABLE / "admission capacity is
#                       temporarily exhausted" / "admission rejected"）
#   err_queue_timeout   master 排队超时 8503（QUEUE_TIMEOUT /
#                       {"status_name":"GatewayTimeout"}）
#   err_rst_stream      gRPC RST_STREAM（HTTP/2 流被远端重置；先于 INTERNAL
#                       匹配，因为 "INTERNAL: RST_STREAM ..." 同时含两者）
#   err_goaway          gRPC GOAWAY（连接关闭；先于 UNAVAILABLE，因为
#                       "UNAVAILABLE: GOAWAY received" 同时含两者）
#   err_unavailable     gRPC UNAVAILABLE（io exception / Network closed）
#   err_cancelled       gRPC CANCELLED（含 REQUEST_CANCELLED 8504；客户端
#                       DEADLINE_EXCEEDED 文本虽含 closed=[CANCELLED] 片段，
#                       但已被上游 err_deadline（小写 "deadline"，命中
#                       "CallOptions deadline exceeded"）先捕获，不会误入）
#   err_internal        gRPC INTERNAL（引擎侧内部错误）
#   err_empty_response  流完成零输出（status=empty_response / "zero outputs"）
#   err_duplicate_rid   "duplicate request_id: N"（replay 前缀未生效的基建信号）
#   err_no_prefill      master 准入层 "NO_PREFILL_WORKER"（8400 族 P 侧无
#                       可用 worker；20P 小集群排水不足时主导拒绝，750P
#                       时代罕见故 20260828 补桶；与 err_no_decode 同族）
# 匹配不上任何具名桶的才落 err_other 残渣（如 BATCH_SLO_EXPIRED 等 enum
# 名、纯 gRPC "UNKNOWN"、"interrupted"、纯码 "code=84xx"）。
# 8xxx 码匹配要求 code= 前缀（JavaLoadClient 纯码回退恒为 "code=NNNN"
# 形态，行 477-479），避免误命中时间戳/地址端口里的连续数字子串。
ERR_ADMISSION_CODE_RE = re.compile(r"code\s*[=:]\s*843[012]\b")
ERR_QUEUE_TIMEOUT_CODE_RE = re.compile(r"code\s*[=:]\s*8503\b")


def classify_error(status, err):
    """Error row -> named bucket key (priority chain, see block comment)."""
    # ---- 既有具名桶（匹配规则与旧版逐字一致，仅提取成函数） ----
    if "preempted by higher-priority" in err or "8429" in err:
        # Auto-TPM eviction terminal (code=8429).
        return "err_preempted"
    if "yielded to higher-priority" in err:
        # Auto-TPM yielded terminal (carried on retryable 8400).
        return "err_yielded"
    if "requests are ahead" in err:
        # Priority-admission queueing rejection ("higher/same-priority
        # requests are ahead"), the fixed-window overload terminal.
        return "err_priority"
    if "NO_DECODE_WORKER" in err or "NO_AVAILABLE_WORKER" in err:
        return "err_no_decode"
    if "NO_PREFILL_WORKER" in err:
        # 8400 族 P 侧无可用 worker（20P 小集群排水不足时主导拒绝）。
        return "err_no_prefill"
    if "queue full" in err or "Worker scheduling queue rejected" in err:
        # 后者：RequestLifecycleCoordinator offer 拒绝（batcher 队列满），
        # 实测文本无 ": queue full" 后缀，需单独子串。
        return "err_queue_full"
    if "SLO expired" in err or "deadline" in err:
        # 8511 SLO expired + gRPC DEADLINE_EXCEEDED（文本含小写
        # "CallOptions deadline exceeded"）+ watchdog "response deadline
        # exceeded"。
        return "err_deadline"
    # ---- err_other 细分子桶（新增；全部匹配不上才落残渣） ----
    if (
        "admission capacity is temporarily exhausted" in err
        or "admission rejected" in err
        or "RESOURCE_EXHAUSTED" in err
        or "ADMISSION_UNAVAILABLE" in err
        or "PRIORITY_ADMISSION_REJECTED" in err
        or ERR_ADMISSION_CODE_RE.search(err)
    ):
        return "err_backpressure"
    if (
        "QUEUE_TIMEOUT" in err
        or "GatewayTimeout" in err
        or ERR_QUEUE_TIMEOUT_CODE_RE.search(err)
    ):
        return "err_queue_timeout"
    if "RST_STREAM" in err:
        return "err_rst_stream"
    if "GOAWAY" in err:
        return "err_goaway"
    if "UNAVAILABLE" in err:
        return "err_unavailable"
    if "CANCELLED" in err:
        return "err_cancelled"
    if "INTERNAL" in err:
        return "err_internal"
    if status == "empty_response" or "zero outputs" in err:
        return "err_empty_response"
    if "duplicate request_id" in err:
        return "err_duplicate_rid"
    return "err_other"


# ---- Phase A 共享统计原语（consolidate_run_outputs.py 经 sentinel 受限 ----
# ---- exec 复用同一份实现；本块只能依赖其上方定义，不得引用其后名字） ----
# percentile/rate/peak/summary 四族在全仓库统一为 nearest-rank 口径，
# 公式逐字搬 run_online_eval.sh 多 worker 合并段（L1397-1448）与
# JavaLoadClient.LatencySummary 形状。


def percentile_nr(values, p, nd=1):
    """Nearest-rank 分位：int(n*p) 取秩（全仓库统一实现，Phase A）。

    与旧 pct 逐字等价（v[min(n-1, int(n*p))]）；空表返回 0。nd=1 与
    aggregate 旧 pct 精度对齐；pacing 分布用 nd=3（sh 合并段 distribution
    的 round 3——p99 与 limit 比较需保留亚毫秒精度，round 1 会在边界值
    上翻转判定）。
    """
    if not values:
        return 0
    v = sorted(values)
    return round(v[min(len(v) - 1, int(len(v) * p))], nd)


def rank_rate(epoch_ms_values):
    """全程速率：(n-1)*1000/(max-min)（sh L1397-1401 同式；<2 样本→0）。"""
    if len(epoch_ms_values) < 2:
        return 0.0
    lo, hi = min(epoch_ms_values), max(epoch_ms_values)
    if hi <= lo:
        return 0.0
    return round((len(epoch_ms_values) - 1) * 1000.0 / (hi - lo), 3)


def latency_summary(values, nd=1):
    """LatencySummary 形状：count/p50/p90/p95/p99/max/mean（nearest-rank）。"""
    if not values:
        return {
            "count": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(values),
        "p50": percentile_nr(values, 0.50, nd),
        "p90": percentile_nr(values, 0.90, nd),
        "p95": percentile_nr(values, 0.95, nd),
        "p99": percentile_nr(values, 0.99, nd),
        "max": round(max(values), nd),
        "mean": round(sum(values) / len(values), nd),
    }


# ---- shared-impl-end: exec 切块到此为止（下方数据加载不进共享段） ----


# ---- inputs: consolidated run-root layout (the only supported form) ----
# 当前格式（consolidate 后）：client.json（server_latency / slo 嵌入）+
# run 根 client_events.jsonl(.gz) + engine_events.jsonl(.gz) + master.json +
# mock.json。legacy load_client/summary.json 双源切换与 shard_* 布局读取链
# 已删（no-backward-compat，旧 run 不再是支持对象）；summary.json 文件随
# Phase B 一并消失（client 只记原始 rows），aggregate 的全部派生统计
# 自 rows 计算，元数据键（sent_task_count / load_client_workers）同样
# rows / client.json 自算，不再从任何 client 侧 summary 透传。
# master.json / mock.json 的 timeseries 与快照降级为对比源（与 jsonl
# rid-join 主数据源互为印证），不再承担 per-request 数据职责。
client_json = load_json("client.json") or {}
summary = client_json
slo = client_json.get("slo_batch_analysis") or {}

# run_meta.json（params + client_env）：Phase A 派生统计需要 client_env 里的
# CLIENT_PACING_LAG_P99_LIMIT_MS，提前到此处加载（原先在 compact time series
# 段才读；纯 load，无副作用）。
run_meta = load_json("run_meta.json") or {}

# ---- per_second from client_events.jsonl (bucket by wall-clock send time) ----
# run 根合并文件（plain 或 gzip）唯一来源；shard_* 与 load_client/ 布局
# 链已删（consolidate 成功即删 shard，残留仅意味着旧 run 复用，不支持）。
# fail-closed：文件缺失直接报错退出（no-backward-compat：jsonl 数据层
# 之前的旧 run 不再支持，不允许静默空 rows 走 None 派生链）。
_client_events_candidates = [
    n for n in ("client_events.jsonl", "client_events.jsonl.gz") if os.path.isfile(n)
]
if not _client_events_candidates:
    sys.exit(
        "ERROR: client_events.jsonl / client_events.jsonl.gz not found in run dir "
        "(multi-component JSONL data layer is the only supported input; "
        "runs older than the jsonl migration are no longer supported)"
    )
rows = []
for f in _client_events_candidates:
    opener = gzip.open if f.endswith(".gz") else open
    with opener(f, "rt", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                sys.exit(f"ERROR: malformed JSON row in {f}: {line[:200]!r}")
if not rows:
    sys.exit(
        "ERROR: client_events.jsonl parsed to zero rows — an empty stream is "
        "not a valid run; refusing to emit an all-None aggregate"
    )
# epoch0 anchor: min over rows that ACTUALLY carry send_start_epoch_ms.
# The old `d.get(..., 0)` default polluted the min with zeros whenever any
# row lacked the field, shifting every rebased series earlier; rows with no
# value are now skipped, and 0 is kept only when NO row has a value.
_send_ts_values = [
    d["send_start_epoch_ms"] for d in rows if d.get("send_start_epoch_ms")
]
epoch0 = min(_send_ts_values) if _send_ts_values else 0
# Rows whose send_start_epoch_ms exists but is 0/None (e.g. fast-rejected
# priority-0 requests that never stamped the field) cannot be placed on the
# time axis: bucketing them would land at a bogus huge-negative t (0-epoch0)
# or, if defaulted to epoch0, spike the t=0 bucket. They are counted and
# surfaced through the integrity marker instead.
per_second_unstamped = 0
error_breakdown = defaultdict(int)

# ---- Phase A 派生统计原始样本（单遍收集；公式见 sh 多 worker 合并段） ----
# server_latency 终态双路径：多 worker 合并段写 load_client/server_latency.json
# （consolidate 保留在原地），consolidate 布局则嵌入 client.json；两条都试。
# 均缺失（旧 run / 单 worker 未采）→ 空 dict，validity 对应检查项按
# “数据缺失”（None）标注而非误报失败。
server_latency = load_json("load_client/server_latency.json")
if not isinstance(server_latency, dict) or not server_latency:
    _embedded_sl = client_json.get("server_latency")
    server_latency = dict(_embedded_sl) if isinstance(_embedded_sl, dict) else {}
rpc_start_ms = []  # stamped 行 send_start（actual_send_qps 轴）
pacing_lag_samples = []  # stamped 行 pacing_lag（Java 同规则：仅 send_start>0）
ttft_samples = []  # is_ok 行 ttft_ms>0（全程分位，全量口径）
e2e_samples = []  # is_ok 行 total_ms>0
sched_client_samples = []  # is_ok 行 schedule_ms（schedule 双源的 client 口径）
ok_count = 0  # is_ok 行数（success_count 自算口径）
completed_count = 0  # is_ok 行数（completed 口径；Java 成功态 "scheduled" / 旧 "ok"）
ok_input_tokens = 0  # is_ok 行 Σinput_len（input_token_tps 分子，完成请求口径）
ok_output_tokens = 0  # is_ok 行 Σoutput_len（output_token_tps 分子，同口径）
wall_clock_vals = []  # wall_clock_ts（秒）——elapsed_s 主口径窗口

# ---- engine per-rid terminal rows: full_e2e + birth-axis exec join ----
# JavaMockEngineCluster streams one JSONL row per engine-side terminal into
# engine_events.jsonl (run root, plain or gzip, ONE shared file for all
# engines): event=decode_done carries rid / engine_name / batch_id /
# engine_arrival_ms / decode_start_ms / decode_done_ms / exec_ms /
# batch_size / output_len / kv_used_tokens / cancelled; event=prefill_done
# carries the prefill twin set (prefill_start_ms / prefill_done_ms / ttft_ms
# / input_len / cache_hit_tokens; exec_ms = BATCH duration — prefill runs
# whole batches). Client, master and the mock engine all run in the same
# container, so decode_done_ms / prefill_done_ms and the client's
# send_start_epoch_ms share one wall clock — no domain conversion needed.
# cancelled=true rows are non-normal terminals and are skipped (they never
# join; the row lands in the join-miss integrity markers instead of being
# fabricated). Two consumers share one pass over the engine stream:
#   * full_e2e (schedule-only full path): decode_done_ms - send_start
#   * birth-axis engine exec percentiles: exec_ms bucketed by the request's
#     BIRTH second (send_start) — same axis as e2e/full_e2e, unlike the
#     legacy engine_exec_ts completion-window snapshot (kept unchanged).
# fail-closed: this stream is a peer of client_events.jsonl — a missing
# file, a malformed row, an unknown event kind, or a row missing rid /
# terminal ts / exec_ms is a hard error, never a silent empty full_e2e
# column (no-backward-compat: legacy grep-the-stdout runs are no longer
# supported).
decode_done_map = {}
prefill_done_map = {}
full_e2e_join_miss = 0
prefill_exec_join_miss = 0
_engine_events_candidates = [
    n for n in ("engine_events.jsonl", "engine_events.jsonl.gz") if os.path.isfile(n)
]
if not _engine_events_candidates:
    sys.exit(
        "ERROR: engine_events.jsonl / engine_events.jsonl.gz not found in run dir "
        "(multi-component JSONL data layer is the only supported input; "
        "runs older than the jsonl migration are no longer supported)"
    )
for _evf in _engine_events_candidates:
    _ev_opener = gzip.open if _evf.endswith(".gz") else open
    with _ev_opener(_evf, "rt", errors="replace") as _ev_stream:
        for line in _ev_stream:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                sys.exit(f"ERROR: malformed JSON row in {_evf}: {line[:200]!r}")
            try:
                _ev_rid = int(ev["rid"])
            except (KeyError, TypeError, ValueError):
                sys.exit(
                    f"ERROR: engine event row missing/invalid rid in {_evf}: "
                    f"{line[:200]!r}"
                )
            if ev.get("cancelled"):
                continue
            _ev_kind = ev.get("event")
            try:
                if _ev_kind == "decode_done":
                    decode_done_map[_ev_rid] = (
                        int(ev["decode_done_ms"]),
                        int(ev["exec_ms"]),
                    )
                elif _ev_kind == "prefill_done":
                    prefill_done_map[_ev_rid] = (
                        int(ev["prefill_done_ms"]),
                        int(ev["exec_ms"]),
                    )
                else:
                    sys.exit(
                        f"ERROR: unknown engine event kind {_ev_kind!r} in {_evf}: "
                        f"{line[:200]!r}"
                    )
            except (KeyError, TypeError, ValueError):
                sys.exit(
                    f"ERROR: {_ev_kind} row missing/invalid terminal ts or exec_ms "
                    f"in {_evf}: {line[:200]!r}"
                )
per_sec = defaultdict(
    lambda: {
        "arrivals": 0,
        "success": 0,
        "errors": 0,
        "err_no_decode": 0,
        "err_no_prefill": 0,
        "err_queue_full": 0,
        "err_deadline": 0,
        "err_priority": 0,
        "err_preempted": 0,
        "err_yielded": 0,
        "err_backpressure": 0,
        "err_queue_timeout": 0,
        "err_rst_stream": 0,
        "err_goaway": 0,
        "err_unavailable": 0,
        "err_cancelled": 0,
        "err_internal": 0,
        "err_empty_response": 0,
        "err_duplicate_rid": 0,
        "err_other": 0,
        "sched": [],
        "e2e": [],
        "ttft": [],
        "full_e2e": [],
        "prefill_exec": [],
        "decode_exec": [],
        "input_len": [],
        "output_len": [],
        "input_tokens": 0,
        "output_tokens": 0,
        # output 完成口径时序（20260901 修复）：ok 行按完成秒分桶的
        # Σol，与 mock 自报 rtp_llm_generate_tps 的完成事件记账同轴对账
        # （出生口径 output_tokens 在 ramp/drain 期因在途请求错位）。桶
        # 键集合 = 出生秒 ∪ 完成秒（drain 期在途完成会新建纯完成桶）。
        "output_tokens_completed": 0,
    }
)
for d in rows:
    _send_ts = d.get("send_start_epoch_ms")
    err = d.get("error") or ""
    # 全量错误构成统计（含无时间戳行，与 summary.error_count 同口径；
    # per_second 只统计带时间戳行）。
    _bucket_key = classify_error(d.get("status"), err) if not is_ok(d) else None
    if _bucket_key is not None:
        error_breakdown[_bucket_key] += 1
    else:
        # Phase A 全程样本（全量口径，与 error_breakdown 同级：含无时间戳
        # is_ok 行）。schedule_ms 缺省 0 与 per_second 桶化同规则。
        ok_count += 1
        # 完成 token 口径（20260901）：input/output_token_tps 的分子，
        # 与 mock 自报 TPS 的未取消完成事件语义对齐（被拒/取消请求的
        # token 不进完成吞吐）。
        ok_input_tokens += d.get("input_len") or 0
        ok_output_tokens += d.get("output_len") or 0
        # Bug #2 修复：completed 与 is_ok 成功口径对齐。原判 status=="ok"
        # 只覆盖 sh 合并段的旧形态；Java load client 成功态写 "scheduled"
        # （无 error 键），旧判使 completed_count/completed_qps 恒 0。此处
        # 位于 _bucket_key is None 分支（即 is_ok(d) 为真），与全聚合器
        # 唯一成功谓词一致：scheduled / ok 两种形态都计入。
        completed_count += 1
        sched_client_samples.append(d.get("schedule_ms", 0))
        if d.get("total_ms"):
            e2e_samples.append(d["total_ms"])
        if d.get("ttft_ms"):
            ttft_samples.append(d["ttft_ms"])
    if d.get("wall_clock_ts"):
        wall_clock_vals.append(d["wall_clock_ts"])
    if not _send_ts:
        per_second_unstamped += 1
        continue
    # 时间轴样本仅 stamped 行（sh 合并段 L1273-1278 同规则：
    # send_start<=0 跳过——合成 timeout/exception 行无该键；pacing_lag
    # 无值时记 0，与 sh 逐字一致）。
    rpc_start_ms.append(float(_send_ts))
    pacing_lag_samples.append(float(d.get("pacing_lag_ms", 0.0) or 0.0))
    t = int((_send_ts - epoch0) // 1000)
    b = per_sec[t]
    b["arrivals"] += 1
    # token 形状时序（20260901）：input_len/output_len 按出生秒分桶，
    # 全部带时间戳请求行（含错误行——形状刻画的是输入组成，与幸存者
    # 口径的 sched/e2e 不同）；缺字段行跳过，桶样本数走 input_len_n。
    if d.get("input_len") is not None:
        b["input_len"].append(d["input_len"])
    if d.get("output_len") is not None:
        b["output_len"].append(d["output_len"])
    # token 吞吐时序（20260901，出生秒分桶，全部带时间戳行）：client
    # 到达口径 Σil/Σol，报告层与 mock 自报 rtp_llm_*（完成事件记账）
    # 对账，差值 = 排队/拒绝吃掉的部分；错误行 output_len 缺省按 0
    # 累加（未产出的 token 不计入到达流量）。
    b["input_tokens"] += d.get("input_len") or 0
    b["output_tokens"] += d.get("output_len") or 0
    if _bucket_key is None:
        # output 完成口径时序（20260901 修复）：ok 行按完成秒分桶累加
        # ol。完成时刻 = send_start + total_ms（total_ms 终点是最后一帧
        # finished，与引擎侧 decode 完成回调同点；不用 wall_clock_ts——
        # 它在流消费完之后才取且 scheduled 路径缺赋值，覆盖不完整）。
        # total_ms 缺失的行跳过（计入与 ok_output_tokens 的差值，不编造
        # 分桶）；完成秒可超出出生窗尾（drain 期在途完成），defaultdict
        # 自动新建纯完成桶（arrivals=0，仅 output_tokens_completed 非零）。
        if d.get("total_ms"):
            _tc = int((_send_ts + d["total_ms"] - epoch0) // 1000)
            per_sec[_tc]["output_tokens_completed"] += d.get("output_len") or 0
        b["success"] += 1
        b["sched"].append(d.get("schedule_ms", 0))
        if d.get("total_ms"):
            b["e2e"].append(d["total_ms"])
        if d.get("ttft_ms"):
            b["ttft"].append(d["ttft_ms"])
        # full_e2e（schedule-only 全链路）：client 发出 → 引擎侧 decode
        # 正常终态，按 request_id（引擎 GenerateInputPB.requestId 原样
        # 回传的数值字段）关联。ok 行 join 不到终态行（run 结束仍在
        # decode 的 in-flight、cancelled 终态、jsonl 截断）或关联出
        # 负值（时钟异常）时计 miss 不编造；全部终态行均 cancelled
        # （map 空，理论上仅极端场景）时整列不产出、也不计 miss。
        # 同一 join 顺带产出出生轴 decode_exec：终态行 exec_ms 归入该
        # 请求的出生秒桶（与 e2e/full_e2e 同轴可比；旧完成轴快照见
        # engine_exec_ts，字段保留）。decode 侧 join miss 与 full_e2e
        # 同源同数（full_e2e_join_miss），不重复计数。
        _rid = d.get("request_id")
        try:
            _rid = int(_rid)
        except (TypeError, ValueError):
            _rid = None
        if decode_done_map:
            _done = decode_done_map.get(_rid) if _rid is not None else None
            if _done is not None and _done[0] >= _send_ts:
                b["full_e2e"].append(_done[0] - _send_ts)
                b["decode_exec"].append(_done[1])
            else:
                full_e2e_join_miss += 1
        # prefill_exec（出生轴）：ok 行按 rid join 引擎侧 prefill 批完成
        # 行（exec_ms = 批执行时长，同批成员同值）。join 不到（cancelled
        # 批成员/jsonl 截断/时钟异常）计 prefill_exec_join_miss 不编造；
        # 全部终态行均 cancelled（map 空）时不产出不计 miss。
        if prefill_done_map:
            _pf = prefill_done_map.get(_rid) if _rid is not None else None
            if _pf is not None and _pf[0] >= _send_ts:
                b["prefill_exec"].append(_pf[1])
            else:
                prefill_exec_join_miss += 1
    else:
        b["errors"] += 1
        b[_bucket_key] += 1


def pct(v, p):
    # Phase A 统一：per_second 桶分位改走共享 nearest-rank 实现（逐字等价）。
    return percentile_nr(v, p)


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
            "err_no_prefill": b["err_no_prefill"],
            "err_queue_full": b["err_queue_full"],
            "err_deadline": b["err_deadline"],
            "err_priority": b["err_priority"],
            "err_preempted": b["err_preempted"],
            "err_yielded": b["err_yielded"],
            "err_backpressure": b["err_backpressure"],
            "err_queue_timeout": b["err_queue_timeout"],
            "err_rst_stream": b["err_rst_stream"],
            "err_goaway": b["err_goaway"],
            "err_unavailable": b["err_unavailable"],
            "err_cancelled": b["err_cancelled"],
            "err_internal": b["err_internal"],
            "err_empty_response": b["err_empty_response"],
            "err_duplicate_rid": b["err_duplicate_rid"],
            "err_other": b["err_other"],
            "sched_p50": pct(b["sched"], 0.5),
            "sched_p95": pct(b["sched"], 0.95),
            "sched_p99": pct(b["sched"], 0.99),
            # e2e_n：与 e2e 分位同源的每秒样本数（幸存者口径，只有
            # status=ok 且带 total_ms 的行才进 e2e 列表）。
            "e2e_n": len(b["e2e"]),
            "e2e_p50": pct(b["e2e"], 0.5),
            "e2e_p95": pct(b["e2e"], 0.95),
            "ttft_p50": pct(b["ttft"], 0.5),
            "ttft_p95": pct(b["ttft"], 0.95),
            # full_e2e：跨两侧全链路（发出→decode 结束）分位，样本为
            # ok 行 join 到引擎终态行的差值（见桶化处注释）。
            "full_e2e_n": len(b["full_e2e"]),
            "full_e2e_p50": pct(b["full_e2e"], 0.5),
            "full_e2e_p95": pct(b["full_e2e"], 0.95),
            # 引擎执行分位（出生轴，20260830）：ok 行按 rid join 引擎终态行
            # （engine_events.jsonl 的 prefill_done / decode_done）的
            # exec_ms，按 send_start 出生秒分桶——与 e2e/full_e2e 同轴可比
            # （幸存者口径）；旧完成轴窗口快照见 engine_exec_ts（字段保留，
            # 口径不同：完成流含 cancel、按完成秒分桶）。终态流全
            # cancelled（map 空）时这些键恒为 0/n=0，报告层按全零回退完成轴。
            "prefill_exec_n": len(b["prefill_exec"]),
            "prefill_exec_p50": pct(b["prefill_exec"], 0.5),
            "prefill_exec_p95": pct(b["prefill_exec"], 0.95),
            "decode_exec_n": len(b["decode_exec"]),
            "decode_exec_p50": pct(b["decode_exec"], 0.5),
            "decode_exec_p95": pct(b["decode_exec"], 0.95),
            # token 长度时序（出生秒分桶，全部带时间戳行）：replay run
            # 的长度组成随 trace loop 周期变化，供报告层与 batch size
            # 时序对照（输入侧驱动识别）。旧 run 无字段时桶为空 -> n=0。
            "input_len_n": len(b["input_len"]),
            "input_len_p50": pct(b["input_len"], 0.5),
            "input_len_p95": pct(b["input_len"], 0.95),
            "output_len_n": len(b["output_len"]),
            "output_len_p50": pct(b["output_len"], 0.5),
            "output_len_p95": pct(b["output_len"], 0.95),
            # token 吞吐时序（20260901，出生秒分桶，全部带时间戳行）：
            # client 到达口径 Σil/Σol，报告层与 mock 自报 rtp_llm_*
            # 完成口径对账（差值 = 调度链路损耗）；与 input_len 分位
            # 同数据源不同聚合（求和 vs 分位）。
            "input_tokens": b["input_tokens"],
            "output_tokens": b["output_tokens"],
            # token 完成口径时序（20260901 修复，完成秒分桶 ok 行）：
            # Σoutput_tokens_completed 与 summary.output_token_tps 的分子
            # ok_output_tokens 同源，差值 = 缺 total_ms / 无时间戳的 ok
            # 行（不编造分桶）；供报告层 output 侧完成口径守恒对账。
            "output_tokens_completed": b["output_tokens_completed"],
        }
    )

# full_e2e 全程终态分位（跨桶合并样本）；无样本（旧引擎日志/全 miss）时
# 为 None，out.summary 与报告层按缺字段回退。
full_e2e_all = []
for _t in sorted(per_sec):
    full_e2e_all.extend(per_sec[_t]["full_e2e"])
full_e2e_latency_ms = None
if full_e2e_all:
    full_e2e_latency_ms = {
        "count": len(full_e2e_all),
        "p50": pct(full_e2e_all, 0.5),
        "p95": pct(full_e2e_all, 0.95),
        "p99": pct(full_e2e_all, 0.99),
    }

# ---- Phase A 派生统计：validity / quick-stats / 全程分位（统一聚合侧） ----
# 公式逐字搬 run_online_eval.sh 多 worker 合并段（L1253-1362）。rows 是
# 唯一指标源（no-backward-compat；client_events.jsonl 已 fail-closed，
# rows 恒非空，_have_rows 仅保留为防御）；仅 schedule 双源的 server 口径
# 与 server_* 直读键不依赖 rows（server_latency 是当前格式正式输入）。
_have_rows = bool(rows)
_actual_rpc_start_count = len(rpc_start_ms)  # sh: sum(shard actual_sent_count)
_recorded_result_count = len(rows)  # sh: sum(shard recorded_result_count)
# sent_task_count = rows 自算（Phase B）：JavaLoadClient 每个 dispatch 的
# 任务终态必落一行（handleRequest 全路径返回 result，
# collectOutstandingResults 为 outstanding future 合成行），len(rows)
# 即发送循环 dispatch 总数（旧 SH 段 sum(sent_count) 的等价口径）。
_sent_task_count = len(rows)
# load_client_workers = rows 布局自算（Phase B）：consolidate 在 client.json
# 的 per_request_source.shard_count 记录实际分片数（单 worker 文件 = 1，
# 多 worker = shard 目录数，与旧 worker_count 同口径）；re-run 恢复模式
# （=0）回落 run_meta params 的配置值。
_prs = client_json.get("per_request_source") or {}
_shard_count = _prs.get("shard_count")
_load_client_workers = (
    _shard_count
    if isinstance(_shard_count, int) and _shard_count > 0
    else run_meta.get("params", {}).get("load_client_workers")
)
_error_count_calc = len(rows) - ok_count  # 与 error_breakdown 同口径（全量行）
_success_count_calc = ok_count
# pacing 分布：sh distribution 的 round 3 精度（p99 与 limit 比较保真）。
_pacing_dist = latency_summary(pacing_lag_samples, nd=3)
_ttft_summary_calc = latency_summary(ttft_samples)
_e2e_summary_calc = latency_summary(e2e_samples)

# pacing limit：client_env 快照（run_meta.client_env，字符串值）；缺省
# 100.0（run_online_eval.sh 的 CLIENT_PACING_LAG_P99_LIMIT_MS 默认值）。
_pacing_limit_raw = (run_meta.get("client_env") or {}).get(
    "CLIENT_PACING_LAG_P99_LIMIT_MS"
)
try:
    pacing_limit_ms = float(_pacing_limit_raw)
except (TypeError, ValueError):
    pacing_limit_ms = 100.0

# elapsed_s 主口径 wall_clock_ts 窗口（秒）；无 wall_clock 行回退发送窗口。
if wall_clock_vals:
    elapsed_s_calc = round(max(wall_clock_vals) - min(wall_clock_vals), 3)
elif len(rpc_start_ms) >= 2:
    elapsed_s_calc = round((max(rpc_start_ms) - min(rpc_start_ms)) / 1000.0, 3)
else:
    elapsed_s_calc = None

# schedule 双源裁决：server_total_ms 有样本（count>0）→ server 口径（master
# 侧算好的分位对象，透传不重算）；否则 rows 有 ok 样本 → client 口径
# （本层 nearest-rank 重算）；两者皆无 → None（透传回退）。
_server_total = server_latency.get("server_total_ms")
_schedule_latency_calc = None
_schedule_source_calc = None
if isinstance(_server_total, dict) and _server_total.get("count"):
    _schedule_latency_calc = dict(_server_total)
    _schedule_source_calc = "server"
elif sched_client_samples:
    _schedule_latency_calc = latency_summary(sched_client_samples)
    _schedule_source_calc = "client"
# server 五阶段延迟（sh L1355-1358 同键集；server 缺某阶段 → 空 dict 透传）
_server_stage_calc = None
if server_latency:
    _server_stage_calc = {
        key: (server_latency.get(key) or {})
        for key in (
            "grpc_queue_ms",
            "route_submit_ms",
            "batch_wait_ms",
            "dispatch_ack_ms",
            "ack_response_ms",
        )
    }

# validity 六项（sh L1315-1322 语义；缺输入 → None = 数据缺失标注，
# test_valid 保守判 invalid：None 不是 True）。与 sh 的两处有意差异：
# master_arrival/completion 的 server 侧计数缺失时 sh 当 0 比较（必 False），
# 本层改为 None（旧 run 无 server_latency.json 时不误报“失败”）。
if _have_rows:
    validity_checks_calc = {
        "zero_errors": _error_count_calc == 0,
        "all_scheduled_tasks_started": (
            _sent_task_count == _actual_rpc_start_count
            if _sent_task_count is not None
            else None
        ),
        "all_started_rpcs_recorded": (
            _actual_rpc_start_count == _recorded_result_count
        ),
        "master_arrival_matches_success": (
            server_latency.get("arrival_count") == _success_count_calc
            if server_latency.get("arrival_count") is not None
            else None
        ),
        "master_completion_matches_success": (
            server_latency.get("completion_count") == _success_count_calc
            if server_latency.get("completion_count") is not None
            else None
        ),
        "client_pacing_p99_within_limit": (
            _pacing_dist["p99"] <= pacing_limit_ms if pacing_lag_samples else None
        ),
    }
    test_valid_calc = all(v is True for v in validity_checks_calc.values())
else:
    validity_checks_calc = None
    test_valid_calc = None

# quick-stats 族（rows 非空才有意义；空 rows → None，no-backward-compat 无透传）
if _have_rows:
    actual_send_qps_calc = rank_rate(rpc_start_ms)
    _es = elapsed_s_calc or 0.0
    success_qps_calc = round(ok_count / _es, 3) if ok_count and _es else 0.0
    error_qps_calc = (
        round(_error_count_calc / _es, 3) if _error_count_calc and _es else 0.0
    )
    completed_qps_calc = (
        round(completed_count / _es, 3) if completed_count and _es else 0.0
    )
    # client 侧 token 吞吐（完成请求口径，20260901）：ok 行 Σil/Σol ÷
    # elapsed——与 mock 自报 rtp_llm_*（未取消完成事件记账）语义对齐，
    # 是「TPS 容量用带 cache 口径评估」的 client 侧读数。
    input_token_tps_calc = (
        round(ok_input_tokens / _es, 3) if ok_input_tokens and _es else 0.0
    )
    output_token_tps_calc = (
        round(ok_output_tokens / _es, 3) if ok_output_tokens and _es else 0.0
    )
    # error_rate：sh L1363-1365 同式（round 6；分母 recorded 口径）。
    error_rate_calc = (
        round(_error_count_calc / _recorded_result_count, 6)
        if _recorded_result_count
        else 0.0
    )
else:
    actual_send_qps_calc = None
    success_qps_calc = None
    error_qps_calc = None
    completed_qps_calc = None
    input_token_tps_calc = None
    output_token_tps_calc = None
    error_rate_calc = None

# ---- queue_timeseries from java_mock_stats (legacy log first, mock.json) ----
mock_payload = load_json("mock.json") or {}
# cache_saved_tokens：KV 复用收益量化（20260901）——引擎 final_snapshot
# 的累计 hit_tokens_total（getSnapshot 累计键，永不 drain；consolidate
# live fetch 落 mock.json）跨引擎求和。与 mock 自报 context_tps_with_cache
# − context_tps 的窗口差值互为印证（累计总量 vs 瞬时速率）。旧 run 无
# 该键 → None（数据缺失标注，不误报 0）。
cache_saved_tokens_calc = None
_fs_snapshot = (
    mock_payload.get("final_snapshot") or {} if isinstance(mock_payload, dict) else {}
)
if isinstance(_fs_snapshot, dict) and isinstance(_fs_snapshot.get("engines"), list):
    _hit_total = 0
    _hit_seen = False
    for _engine_snap in _fs_snapshot["engines"]:
        if isinstance(_engine_snap, dict) and isinstance(
            _engine_snap.get("hit_tokens_total"), (int, float)
        ):
            _hit_total += _engine_snap["hit_tokens_total"]
            _hit_seen = True
    if _hit_seen:
        cache_saved_tokens_calc = _hit_total
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
_raw_ts = []
for kv in mock_stats:
    ts = int(float(kv.get("ts_epoch_ms", 0)))
    if t0 is None:
        t0 = ts
    _raw_ts.append(ts)
    queue_ts.append(
        {
            "t_offset_s": round((ts - t0) / 1000),
            "prefill_waiting": int(float(kv.get("prefill_waiting", 0))),
            "prefill_running": int(float(kv.get("prefill_running", 0))),
            "prefill_running_reqs": int(float(kv.get("prefill_running_reqs", 0))),
            "max_prefill_waiting": int(float(kv.get("max_prefill_waiting", 0))),
            "decode_waiting": int(float(kv.get("decode_waiting", 0))),
            "decode_running": int(float(kv.get("decode_running", 0))),
            "decode_run_min": int(float(kv.get("decode_run_min", 0))),
            "decode_run_max": int(float(kv.get("decode_run_max", 0))),
            "max_decode_waiting": int(float(kv.get("max_decode_waiting", 0))),
            "cum_prefill_batches": int(float(kv.get("prefill_batches", 0))),
            "cum_enqueued_requests": int(float(kv.get("enqueued_requests", 0))),
            "cum_avg_batch_size": float(kv.get("avg_batch_size", 0)),
            "heap_used_mb": int(float(kv.get("heap_used_mb", 0))),
        }
    )

# ---- t_offset_s rebase to client epoch0（口径修复 20260828）----
# 原锚是 mock 首样本 t0：引擎启动空转期（约 28s）垫在 TQ 轴前段，与
# TSEC / master / queue_top_bottom 序列（epoch0 锚）不同轴——Prefill
# 队列图前段四线贴地重叠、x 轴「压测时间」语义失真。改锚聚合早期从
# client_events send_start_epoch_ms 算出的 epoch0：t = round((ts-epoch0)/1000)，
# 负值样本（早于首个请求发送 = 启动空转）丢弃，首点从 ~0 起；epoch0 == 0
# （无带时间戳行的防御路径）保持旧 mock-t0 锚。interval 差分在
# rebase 之后跑：被丢弃的空转样本 cum 计数为 0，首个保留样本的增量
# 口径不受影响。
if epoch0 and queue_ts:
    _rebased = []
    for _q, _ts in zip(queue_ts, _raw_ts):
        _t_new = round((_ts - epoch0) / 1000)
        if _t_new < 0:
            continue
        _q["t_offset_s"] = _t_new
        _rebased.append(_q)
    if _rebased:
        queue_ts = _rebased

# per-interval batch rate / incremental avg batch size from cumulative counters
prev_b, prev_r = 0, 0
for q in queue_ts:
    db = q["cum_prefill_batches"] - prev_b
    dr = q["cum_enqueued_requests"] - prev_r
    q["interval_batches"] = db
    q["interval_avg_batch_size"] = round(dr / db, 2) if db > 0 else 0
    prev_b, prev_r = q["cum_prefill_batches"], q["cum_enqueued_requests"]

# ---- cancel / decode admission per-second rates (epoch-aligned) ----------
# cancel_rpcs / decode_admitted are cluster CUMULATIVE counters on the
# java_mock_stats line; differencing adjacent samples and normalizing by the
# sample interval yields per-second rates directly on the absolute-epoch
# axis (every line already carries ts_epoch_ms). decode_done is NOT
# cumulative: it is a drained per-tick interval count (DecodeWindow is reset
# on every stats tick), so its per-second rate is the raw value normalized by
# the interval — differencing it would produce a meaningless second-order
# delta. decode_admitted only exists on builds >= this change; older runs
# keep a flat zero series instead of fabricating data.
cancel_qps_ts = []
_stats_anchor = epoch0
if not _stats_anchor:
    for kv in mock_stats:
        try:
            _v = int(float(kv.get("ts_epoch_ms", 0) or 0))
        except (TypeError, ValueError):
            continue
        if _v:
            _stats_anchor = _v
            break

# ---- cancel 按角色拆分（master / prefill / decode 三条速率线） --------
# cancel_rpcs 的完备分解是 census 四项（mock 集群累计计数）：
#   * unknown / finished / tombstone：cancel RPC 到达引擎时引擎已无该请求
#     的活跃条目（从未认识 / 已终态 / 优先级抢占墓碑重复）——这些取消
#     由 master 调度层发起（queueTimeout/deadline 到期、decode endpoint
#     generation retired 批量取消等，请求未到引擎或已离开）→ 归 master
#     侧，三项合并差分即 master_cancel_qps。
#   * tracked：引擎仍有活跃条目时的真实取消。census tracked 在 prefill
#     直收与 prefill→decode 转发两处递增，时序上无法按角色拆分；但 mock.json
#     final_snapshot.engines[] 携带每引擎 cancelled_rids（终值 rid 列表，
#     登记在被取消请求当时所在的引擎上）→ 按 role 分桶即 prefill/decode
#     的 rid 集合。取消时刻用 master 终态行（flexlb_logs/pv.log* / master.log
#     的 request-completion JSON 行）匹配：优先 requestExpiresAtMs（deadline
#     到期即触发 master 取消，与 census 差分时间分布吻合，实测 38/38 落在
#     t=120-178 vs tracked 差分 t=121-164）；缺失时回退 startTime + latencyMs
#     （终态发布时刻 —— 仅兜底：收尾批量 flush 会把它推到 stats 窗口之外，
#     实测 38/38 同落在 t≈250 而 stats 止于 244）。事件按 stats 采样区间
#     (prev_ts, ts] 归组（count / interval_s），与 census 差分同构。匹配
#     不上时刻的 rid 丢弃并计入 integrity（不编造数据）；final_snapshot
#     缺失时 prefill/decode 线保持全零。
_cancel_role_events = {"prefill": [], "decode": []}
_cancel_role_unmatched = {"prefill": 0, "decode": 0}
_fs_engines = (
    # 防御 final_snapshot 为显式 null（consolidate 在 live snapshot fetch
    # 不可用时写入 null + final_snapshot_source=missing/fallback）：键存在但
    # 值为 None 时 .get("engines") 会壁，归一为空 dict 走无快照路径
    # （旧 bug：部分 run 的 consolidate 布局 mock.json 直接让 aggregate 壁）。
    (mock_payload.get("final_snapshot") or {}).get("engines")
    if isinstance(mock_payload, dict)
    else None
)
if isinstance(_fs_engines, list):
    _pv_ts_by_rid = {}
    _pv_files = sorted(glob.glob("flexlb_logs/pv.log*"))
    if not _pv_files and os.path.isfile("master.log"):
        _pv_files = ["master.log"]
    for _f in _pv_files:
        for _line in open(_f, errors="replace"):
            _i = _line.find("{")
            if _i < 0:
                continue
            try:
                _j = json.loads(_line[_i:])
            except ValueError:
                continue
            _rid = _j.get("requestId")
            if _rid is None or _rid in _pv_ts_by_rid:
                continue
            try:
                _exp = int(float(_j.get("requestExpiresAtMs", 0) or 0))
                _st = int(float(_j.get("startTime", 0) or 0))
                _lat = int(float(_j.get("latencyMs", 0) or 0))
            except (TypeError, ValueError):
                continue
            # 优先 deadline 到期时刻（master 取消触发点，落在 stats 窗口内）；
            # 终态发布时刻仅作兜底（收尾批量 flush 常落在窗口外）。
            _t_ev = _exp or (_st + _lat if _st and _lat else 0)
            if _t_ev:
                _pv_ts_by_rid[_rid] = _t_ev
    for _e in _fs_engines:
        if not isinstance(_e, dict):
            continue
        _role = str(_e.get("role") or "").lower()
        if _role not in _cancel_role_events:
            continue
        for _rid in _e.get("cancelled_rids") or []:
            _ts_ev = _pv_ts_by_rid.get(_rid)
            if _ts_ev:
                _cancel_role_events[_role].append(_ts_ev)
            else:
                _cancel_role_unmatched[_role] += 1
    for _role in _cancel_role_events:
        _cancel_role_events[_role].sort()

_prev_stat_ts = None
_prev_cancel = _prev_admitted = _prev_master_cancel = None
for kv in mock_stats:
    try:
        ts = int(float(kv.get("ts_epoch_ms", 0) or 0))
        cancel_cum = int(float(kv.get("cancel_rpcs", 0) or 0))
        admitted_cum = int(float(kv.get("decode_admitted", 0) or 0))
        done_interval = int(float(kv.get("decode_done", 0) or 0))
        # census 三项（引擎侧无活跃请求的 cancel）合并差分 -> master 侧取消
        master_cancel_cum = (
            int(float(kv.get("cancel_census_unknown", 0) or 0))
            + int(float(kv.get("cancel_census_finished", 0) or 0))
            + int(float(kv.get("cancel_census_tombstone", 0) or 0))
        )
    except (TypeError, ValueError):
        continue
    if not ts:
        continue
    if _prev_stat_ts is not None and ts > _prev_stat_ts:
        interval_s = (ts - _prev_stat_ts) / 1000.0
        # tracked cancel 事件按角色归组到采样区间 (prev_ts, ts]
        pf_n = bisect_right(_cancel_role_events["prefill"], ts) - bisect_right(
            _cancel_role_events["prefill"], _prev_stat_ts
        )
        dc_n = bisect_right(_cancel_role_events["decode"], ts) - bisect_right(
            _cancel_role_events["decode"], _prev_stat_ts
        )
        cancel_qps_ts.append(
            {
                "t": round((ts - _stats_anchor) / 1000.0, 1),
                "epoch_ms": ts,
                "cancel_qps": round((cancel_cum - _prev_cancel) / interval_s, 2),
                "master_cancel_qps": round(
                    (master_cancel_cum - _prev_master_cancel) / interval_s, 2
                ),
                "prefill_cancel_qps": round(pf_n / interval_s, 2),
                "decode_cancel_qps": round(dc_n / interval_s, 2),
                "decode_admitted_qps": round(
                    (admitted_cum - _prev_admitted) / interval_s, 2
                ),
                "decode_done_qps": round(done_interval / interval_s, 2),
            }
        )
    _prev_stat_ts = ts
    _prev_cancel, _prev_admitted = cancel_cum, admitted_cum
    _prev_master_cancel = master_cancel_cum

# ---- flexlb structured log files (shared parser source below) ----
# Legacy flexlb_logs/flexlb.log* first; master.log (the consolidated merge)
# carries the same structured lines as the fallback.
log_files = glob.glob("flexlb_logs/flexlb.log*")
if not log_files and os.path.isfile("master.log"):
    log_files = ["master.log"]

# ---- engine_dist: per-engine routing distribution (from client_events rows) ----
# Two scopes since 20260829: prefill/decode = ok rows only (matching
# JavaLoadClient's loadBalanceSummary, legacy keys kept for compatibility);
# prefill_all/decode_all = every row that carries a placement, i.e. the
# scheduler's true routing decisions including failed/timed-out rows (under
# overload the ok rows are a survivor subset and understate imbalance).
# Mock java_mock_stats is cluster-aggregate only, so per-engine utilization
# and KV time series are not computable here (noted, not fabricated).


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
    # All-placement scope: count every row that names an engine, BEFORE the
    # is_ok filter, so failed-but-placed rows (deadline / empty_response /
    # rst_stream after dispatch ...) still represent one routing decision.
    p_count_all = Counter()
    d_count_all = Counter()
    for d in rows:
        p = d.get("prefill") or ""
        de = d.get("decode") or ""
        if p:
            p_count_all[p] += 1
        if de:
            d_count_all[de] += 1
        if not is_ok(d):
            continue
        if p:
            p_count[p] += 1
            p_tokens[p] += d.get("input_len", 0) or 0
        if de:
            d_count[de] += 1
            d_tokens[de] += d.get("output_len", 0) or 0
        t_ms = d.get("send_start_epoch_ms")
        if t_ms:
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
    p_vals_all = sorted(p_count_all.values(), reverse=True)
    d_vals_all = sorted(d_count_all.values(), reverse=True)
    engine_dist["prefill_all"] = {
        "engine_count": len(p_count_all),
        "requests_per_engine": p_vals_all,
        "total": sum(p_vals_all),
        "gini_cum": gini_coeff(p_vals_all),
        "cv": cv_coeff(p_vals_all),
    }
    engine_dist["decode_all"] = {
        "engine_count": len(d_count_all),
        "requests_per_engine": d_vals_all,
        "total": sum(d_vals_all),
        "gini_cum": gini_coeff(d_vals_all),
        "cv": cv_coeff(d_vals_all),
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
        "prefill_all_y_pct": lorenz_pct(p_vals_all),
        "decode_all_y_pct": lorenz_pct(d_vals_all),
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

# ---- compact time series: G3/G4/G5 + log rows, rebased to epoch0 ----------
# All new series share one time axis: seconds since the first per-request
# send (epoch0). Negative t = pre-send warmup. A series whose source file is
# missing comes out empty; the generator renders charts conditionally.

# run_meta 已在头部加载（Phase A 提前：client_env 的 pacing limit 与
# params 的 fetch_output_stream 双消费点）。
master_json = load_json("master.json") or {}
prom_ts = master_json.get("prometheus_timeseries") or []

# mock per-engine 1s 时序（mock_per_engine_timeseries.json.gz，mock 引擎自身
# 上报的 running / waiting / KV 等 per-engine 指标；缺文件 -> 空表 ->
# 引擎侧 Top/Bottom-5 序列省略）。
mock_per_engine_ts = []
if os.path.isfile("mock_per_engine_timeseries.json.gz"):
    try:
        with gzip.open("mock_per_engine_timeseries.json.gz", "rt") as _f:
            mock_per_engine_ts = json.load(_f) or []
    except (OSError, ValueError):
        mock_per_engine_ts = []


def rel_axis(pts):
    """[(epoch_ms, value)] -> [(t_s, value)] on the per-request send axis.

    Falls back to each series' own first sample when client_events rows are
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


def prom_ts_extract_labeled(base_name, label_name, agg="sum"):
    """G3 prometheus timeline -> {label_value: [(epoch_ms, value)]}.

    Unlike prom_ts_extract (which folds label variants into one series),
    this splits them by one label — e.g. the dispatch reason counter's
    reason="..." tag. Same-(ts,label) variants (extra labels on the same
    base name, e.g. the per-engine series of a reason counter) are folded
    per sample by agg: "sum" for counters (per-engine deltas add up),
    "avg" for per-engine gauges (e.g. the per-dispatch batch size — the
    engine average is the cluster-representative value).
    """
    label_re = re.compile(r"(?:^|,)" + re.escape(label_name) + r'="([^"]*)"(?:,|$)')
    series = {}
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
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            key = str(k)
            name, brace, labels = key.partition("{")
            if name != base_name or not brace:
                continue
            # partition keeps the trailing "}" on the labels fragment —
            # strip it, else a label sitting last (reason="x"}) fails the
            # (?:,|$) anchor in label_re.
            m = label_re.search(labels.rstrip("}"))
            if not m:
                continue
            bucket = series.setdefault(m.group(1), {})
            acc = bucket.get(ts) or [0.0, 0]
            acc[0] += float(v)
            acc[1] += 1
            bucket[ts] = acc
    if agg == "avg":
        return {
            label: sorted((ts, s / c) for ts, (s, c) in points.items())
            for label, points in series.items()
        }
    return {
        label: sorted((ts, s) for ts, (s, c) in points.items())
        for label, points in series.items()
    }


def _ts_role_ip_split(groups, base_name):
    """[{ts, metrics}] timeline -> {role: {engineIp: [(epoch_ms, value)]}}.

    Splits a per-engine metric carrying both role and engineIp tags (e.g.
    app.flexlb.batcher.queue.size / mock_engine_running) by the two labels
    at once. Series missing either label are skipped.
    """
    role_re = re.compile(r'(?:^|,)role="([^"]*)"(?:,|$)')
    # 标签名兼容：master prometheus 用 engineIp，mock per-engine 用 engine_ip
    engine_re = re.compile(r'(?:^|,)(?:engineIp|engine_ip)="([^"]*)"(?:,|$)')
    series = {}
    for grp in groups:
        if not isinstance(grp, dict):
            continue
        metrics = grp.get("metrics")
        if not isinstance(metrics, dict):
            continue
        try:
            ts = float(grp.get("ts", 0) or 0)
        except (TypeError, ValueError):
            continue
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            key = str(k)
            name, brace, labels = key.partition("{")
            if name != base_name or not brace:
                continue
            labels = labels.rstrip("}")
            rm = role_re.search(labels)
            em = engine_re.search(labels)
            if not rm or not em:
                continue
            role_series = series.setdefault(rm.group(1), {})
            role_series.setdefault(em.group(1), []).append((ts, float(v)))
    for role_series in series.values():
        for pts in role_series.values():
            pts.sort(key=lambda p: p[0])
    return series


def prom_ts_extract_role_engine(base_name):
    """G3 prometheus timeline -> {role: {engineIp: [(epoch_ms, value)]}}."""
    return _ts_role_ip_split(prom_ts, base_name)


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

# ---- master_events.jsonl: offline parser over the master logs ------------
# Master production code stays untouched by the multi-component JSONL data
# layer: the master keeps logging plain text, and THIS parser materializes
# the master-side event stream offline (one JSON object per line, written
# to master_events.jsonl in the run dir — the third stream of the data
# layer, produced by the aggregator instead of the component itself):
#   * window rows — one per flexlb_server_schedule_latency snapshot
#     (ServerScheduleLatencyRecorder, 10s cadence, root logger →
#     application.log), full field set incl. count / arrival_qps /
#     completion_qps (the p95-only SERVER_LAT_LINE_RE above feeds
#     stage_latency_ts and stays as-is);
#   * generation rows — worker-topology lifecycle events (EngineSyncRunner,
#     syncLogger → sync.log): generation_created ("Created WorkerStatus
#     generation N for worker: ip:port") and generation_retired
#     ("[remove] retiring missing worker ..." / "[replace] retiring worker
#     topology generation ..."). A steady-state run legitimately has ZERO
#     generation rows.
# Source chain mirrors consolidation: merged master.log first (run-root
# application.log prefix + flexlb.log + sync.log + sync_consistency.log
# sections — BOTH row families live there); the pre-consolidation raw
# layout (flexlb_logs/*, still on disk when aggregate runs standalone)
# as the fallback. fail-closed: no master log at all, or a master log with
# zero window rows, is a hard error (a live master always snapshots).
_MASTER_WINDOW_RE = re.compile(
    r"flexlb_server_schedule_latency count=(\d+) arrival_qps=([\d.]+) "
    r"completion_qps=([\d.]+) server_p50_ms=([\d.]+) server_p95_ms=([\d.]+) "
    r"server_p99_ms=([\d.]+) grpc_queue_p95_ms=([\d.]+) "
    r"route_submit_p95_ms=([\d.]+) batch_wait_p95_ms=([\d.]+) "
    r"dispatch_ack_p95_ms=([\d.]+) ack_response_p95_ms=([\d.]+)"
)
_GENERATION_CREATED_RE = re.compile(
    r"Created WorkerStatus generation (\d+) for worker: ([\w.:-]+)"
)
_GENERATION_RETIRED_RE = re.compile(
    r"\[(remove|replace)\] retiring (?:missing worker|worker topology generation),"
    r" model=([^,]*), role=([^,]*), ipPort=([^,]*), generation=(\d+)"
)
if os.path.isfile("master.log"):
    _master_log_sources = ["master.log"]
else:
    _master_log_sources = sorted(
        glob.glob("flexlb_logs/application.log*")
        + glob.glob("flexlb_logs/flexlb.log*")
        + glob.glob("flexlb_logs/sync.log*")
        + glob.glob("flexlb_logs/sync_consistency.log*")
    ) + (["flexlb.log"] if os.path.isfile("flexlb.log") else [])
if not _master_log_sources:
    sys.exit(
        "ERROR: no master log found in run dir (expected master.log, or "
        "flexlb_logs/{application,flexlb,sync,sync_consistency}.log*, or "
        "run-root flexlb.log) — master_events.jsonl cannot be generated"
    )


def _log_line_epoch_ms(line):
    tm = LOG_TS_RE.match(line)
    if not tm:
        return None
    try:
        return (
            datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            + int(tm.group(2)) / 1000.0
        ) * 1000.0
    except ValueError:
        return None


master_events_rows = []
_master_window_rows = 0
for _mf in _master_log_sources:
    with open(_mf, errors="replace") as _mstream:
        for line in _mstream:
            if "flexlb_server_schedule_latency" in line:
                m = _MASTER_WINDOW_RE.search(line)
                if not m:
                    continue
                ts = _log_line_epoch_ms(line)
                if ts is None:
                    continue
                master_events_rows.append(
                    {
                        "event": "window",
                        "ts_epoch_ms": int(round(ts)),
                        "count": int(m.group(1)),
                        "arrival_qps": float(m.group(2)),
                        "completion_qps": float(m.group(3)),
                        "server_p50_ms": float(m.group(4)),
                        "server_p95_ms": float(m.group(5)),
                        "server_p99_ms": float(m.group(6)),
                        "grpc_queue_p95_ms": float(m.group(7)),
                        "route_submit_p95_ms": float(m.group(8)),
                        "batch_wait_p95_ms": float(m.group(9)),
                        "dispatch_ack_p95_ms": float(m.group(10)),
                        "ack_response_p95_ms": float(m.group(11)),
                    }
                )
                _master_window_rows += 1
            elif "WorkerStatus generation" in line or "retiring" in line:
                ts = _log_line_epoch_ms(line)
                if ts is None:
                    continue
                m = _GENERATION_CREATED_RE.search(line)
                if m:
                    master_events_rows.append(
                        {
                            "event": "generation_created",
                            "ts_epoch_ms": int(round(ts)),
                            "generation": int(m.group(1)),
                            "ip_port": m.group(2),
                        }
                    )
                    continue
                m = _GENERATION_RETIRED_RE.search(line)
                if m:
                    master_events_rows.append(
                        {
                            "event": "generation_retired",
                            "ts_epoch_ms": int(round(ts)),
                            "reason": m.group(1),
                            "model": m.group(2),
                            "role": m.group(3),
                            "ip_port": m.group(4),
                            "generation": int(m.group(5)),
                        }
                    )
if _master_window_rows == 0:
    sys.exit(
        "ERROR: zero flexlb_server_schedule_latency window rows in the master "
        "log(s) — a live master snapshots every "
        "flexlb.server-latency-log-interval-ms (10s default); zero rows means "
        "the master log chain is broken"
    )
master_events_rows.sort(key=lambda r: (r["ts_epoch_ms"], r["event"]))
with open("master_events.jsonl", "w") as _mev_stream:
    for _mev_row in master_events_rows:
        _mev_stream.write(json.dumps(_mev_row, sort_keys=True) + "\n")

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
                ts,
                int(float(kv.get("decode_exec_p50", 0) or 0)),
                int(float(kv.get("decode_exec_p95", 0) or 0)),
                int(float(kv.get("prefill_exec_p50", 0) or 0)),
                int(float(kv.get("prefill_exec_p95", 0) or 0)),
            ),
        )
    )
engine_exec_ts = []
for t, (epoch_ms, d50, d95, p50, p95) in rel_axis(exec_pts):
    row = {
        "t": t,
        "epoch_ms": epoch_ms,
        "decode_exec_p50_ms": d50,
        "decode_exec_p95_ms": d95,
    }
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
# decode 侧 schema 已随 G4 分层准入改版（HttpLoadBalanceServer
# .inflightStatus）：decode_endpoints 旧 inflight_requests 字段已不存在
# （旧键读取恒 0 的 bug 根因），现读 reserved_total（master 预约未
# 确认）与 confirmed_running（引擎确认运行中）；prefill_endpoints 补读
# inflight_requests（master 账本请求数，与 inflight_batches 同源）。
# no-backward-compat：旧 run（旧 schema 快照）不再产出 decode 线。
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
        p_endpoints = [
            e for e in infl.get("prefill_endpoints") or [] if isinstance(e, dict)
        ]
        p_batches = sum(int(e.get("inflight_batches", 0) or 0) for e in p_endpoints)
        p_reqs = sum(int(e.get("inflight_requests", 0) or 0) for e in p_endpoints)
        d_endpoints = [
            e for e in infl.get("decode_endpoints") or [] if isinstance(e, dict)
        ]
        d_reserved = sum(int(e.get("reserved_total", 0) or 0) for e in d_endpoints)
        d_running = sum(int(e.get("confirmed_running", 0) or 0) for e in d_endpoints)
    except (TypeError, ValueError):
        continue
    inflight_pts.append((ts, (sched, p_batches, p_reqs, d_reserved, d_running)))
inflight_ts = [
    {
        "t": t,
        "scheduler": s,
        "prefill_batches": pb,
        "prefill_requests": pr,
        "decode_reserved": drv,
        "decode_confirmed_running": drn,
    }
    for t, (s, pb, pr, drv, drn) in rel_axis(inflight_pts)
]

# master 每秒到达/完成速率（counters_timeseries 累计计数器差分）。
# arrival_count / completion_count 是 master 侧单调累计计数器（1s 采样，
# 间隔 ~1001ms）；相邻样本正差分 ÷ 间隔秒 = 每秒速率（计数器重置的
# 负差分区间丢弃，不造峰）。这是 QPS 图表发送序列的权威数据源：
# 客户端 per_request 的 arrivals 在收集器截断时只覆盖部分窗口
# （实测 A 档 33,372 行止于 0-70s，每秒 ~476），而 master 计数器
# 覆盖全部到达（同 run t=0-120s 每秒 ~2000、累计 239,197、t=121 冻结）。
# rel_axis 重锚 epoch0；负 t 样本（首请求发送前的暖机零值）保留，
# 报告端按需丢弃。rel_axis 的元组透传同时携带 completions 速率与
# 到达累计值（cum_arrivals，冻结点取证用）。
_counter_pts = []
for _row in master_json.get("counters_timeseries") or []:
    if not isinstance(_row, dict):
        continue
    try:
        _ts = int(float(_row.get("ts_epoch_ms", 0) or 0))
        _arr = int(float(_row.get("arrival_count", 0) or 0))
        _cmp = int(float(_row.get("completion_count", 0) or 0))
    except (TypeError, ValueError):
        continue
    if not _ts:
        continue
    _counter_pts.append((_ts, _arr, _cmp))
_arrival_rate_pts = []
for (_ts0, _a0, _c0), (_ts1, _a1, _c1) in zip(_counter_pts, _counter_pts[1:]):
    _dt_s = (_ts1 - _ts0) / 1000.0
    if _dt_s <= 0 or _a1 < _a0:
        continue
    _arrival_rate_pts.append(
        (
            _ts1,
            (
                round((_a1 - _a0) / _dt_s, 1),
                round(max(0, _c1 - _c0) / _dt_s, 1),
                _a1,
            ),
        )
    )
master_arrivals_ts = [
    {
        "t": t,
        "arrivals": v[0],
        "completions": v[1],
        "cum_arrivals": v[2],
    }
    for t, v in rel_axis(_arrival_rate_pts)
]

# master-side queue depth + inflight age from the G3 prometheus timeline
# (needs FLEXLB_MONITOR_MODE=all; per-priority label variants summed).
# ad2d6224+: INFLIGHT_MAX_AGE_MS carries {role, engineIp} tags —
# role=SCHEDULER + engineIp="scheduler" marks the scheduler's own ledger,
# PREFILL/DECODE + real engineIp the per-worker ledgers. Keep age_ms as the
# cluster-wide max (back-compat) and add one max-across-engines series per
# role so the report can draw a line per ledger.
AGE_BASE = "flexlb_app_flexlb_inflight_max_age_ms"
age_pts = prom_ts_extract(AGE_BASE, agg="max")
inflight_age_ts = [{"t": t, "age_ms": int(round(v))} for t, v in rel_axis(age_pts)]
age_role_engine = prom_ts_extract_role_engine(AGE_BASE)
inflight_age_by_role = {}
for role, engines in age_role_engine.items():
    by_ts = {}
    for pts in engines.values():
        for ts, v in pts:
            by_ts[ts] = max(by_ts.get(ts, 0.0), float(v))
    # Bug #1 修复（20260831 run 20260831_134508）：局部变量改名
    # age_rows —— 原名 rows 遮蔽了全局 per_request rows，导致下游
    # summary.total_requests = len(rows) 被污染为 inflight age 时间桶数
    # （实测 261 而非 956179）。
    age_rows = rel_axis(sorted(by_ts.items()))
    if age_rows:
        inflight_age_by_role[role.lower()] = [
            {"t": t, "age_ms": int(round(v))} for t, v in age_rows
        ]

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

# G3 per-engine batcher queue gauge. The metric carries role + engineIp
# tags (BatchSchedulerReporter#reportBatcherQueueSize), so the plain
# prom_ts_extract sum below folds PREFILL + DECODE workers into one
# cluster total — kept for backward compatibility only; the per-role
# totals and per-engine series below carry the real breakdown.
BATCHER_Q_BASE = "flexlb_app_flexlb_batcher_queue_size"
batcher_pts = prom_ts_extract(BATCHER_Q_BASE, agg="sum")
# routing.queue.length's only reporter is reportBatcherQueueDepthByPriority
# (type=batchQueue): the SAME per-engine batcher queue bucketed by priority,
# not an independent routing-stage queue. That priority-bucket view freezes
# its last sample at shutdown (stale tail, an upload artifact — e.g. a
# forever-189 tail instead of draining to 0), so the routing series now
# reuses the batcher_queue_size source: same queue, correct zero-drain.
# Old aggregate JSONs keep their legacy routing_queue values (degraded
# compat): the report caption spells out the active convention.
routing_pts = batcher_pts
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

# Per-role cluster totals + per-engine prefill depth distribution. The
# master exposes no dispatch-time queue-depth counter, so the per-engine
# 1s prometheus samples stand in as the decision-time depth estimate
# (dispatch decisions happen asynchronously on each engine's batcher
# thread; the sampling skew is bounded by the 1s poll interval).
batcher_ts_by_role = []
batcher_top_engines_ts = []
batcher_role_series = prom_ts_extract_role_engine(BATCHER_Q_BASE)
if batcher_role_series:
    role_rows_by_ts = defaultdict(dict)
    for role, engines in batcher_role_series.items():
        by_ts = defaultdict(float)
        for pts in engines.values():
            for ts, v in pts:
                by_ts[ts] += v
        for t, v in rel_axis(sorted(by_ts.items())):
            role_rows_by_ts[t][role.lower()] = int(round(v))
    batcher_ts_by_role = [
        {"t": t, **vals} for t, vals in sorted(role_rows_by_ts.items())
    ]

    prefill_engines = batcher_role_series.get("PREFILL") or {}

    # Top-5 prefill engines by peak depth, downsampled to the last sample
    # per 5s window to keep the aggregate compact.
    engine_peak = {}
    for ip, pts in prefill_engines.items():
        if pts:
            engine_peak[ip] = max(v for _, v in pts)
    top_ips = sorted(engine_peak, key=engine_peak.get, reverse=True)[:5]
    top_by_t = defaultdict(dict)
    for ip in top_ips:
        last_in_bucket = {}
        order = []
        for ts, v in prefill_engines[ip]:
            bucket = int(ts // 5000)
            if bucket not in last_in_bucket:
                order.append(bucket)
            last_in_bucket[bucket] = (ts, v)
        for t, v in rel_axis([last_in_bucket[b] for b in order]):
            top_by_t[t][ip] = round(v, 2)
    batcher_top_engines_ts = [{"t": t, **vals} for t, vals in sorted(top_by_t.items())]

# ---- queue Top/Bottom-5 per-engine series (queue_top_bottom_ts) ----
# 每队列（P master-batcher / P、D 引擎侧 running / waiting）各取按峰值排序
# 的 top-5 与 bottom-5 引擎，5s 窗口降采样，行格式 [{t, "<ip>": v}]。
# top/bottom 同口径（均按峰值），体现负载最重与最轻的引擎；全零引擎
# 也会出现在 bottom-5（线本身就是“饿死”证据）。生成器缺键时退化。


def _downsample_5s(pts):
    """[(epoch_ms, v)] -> 每 5s 窗口取最后样本（保序）。"""
    last_in_bucket = {}
    order = []
    for ts, v in pts:
        bucket = int(ts // 5000)
        if bucket not in last_in_bucket:
            order.append(bucket)
        last_in_bucket[bucket] = (ts, v)
    return [last_in_bucket[b] for b in order]


def _top_bottom_rows(engine_pts, n=5):
    """{ip: [(ts, v)]} -> {"top": rows, "bottom": rows}.

    按峰值排序选 top/bottom 引擎（并列时按 ip 字典序稳定）；行格式
    [{t, "<ip>": v}]（rel_axis 相对秒轴 + 5s 窗口降采样）。
    """
    out = {"top": [], "bottom": []}
    if not engine_pts:
        return out
    ranked = sorted(
        engine_pts,
        key=lambda ip: (max(v for _, v in engine_pts[ip]), ip),
    )
    picks = {
        "top": ranked[-n:][::-1],
        "bottom": ranked[:n],
    }
    for kind, ips in picks.items():
        by_t = defaultdict(dict)
        for ip in ips:
            for t, v in rel_axis(_downsample_5s(engine_pts[ip])):
                by_t[t][ip] = round(v, 2)
        out[kind] = [{"t": t, **vals} for t, vals in sorted(by_t.items())]
    return out


queue_top_bottom_ts = {}

# P master-batcher 队列（master prometheus per-engine 序列，1s 采样；
# 决策时点深度近似）。与上方 batcher_top_engines_ts 同源，但补齐 bottom-5。
if batcher_role_series:
    _tb = _top_bottom_rows(batcher_role_series.get("PREFILL") or {})
    if _tb["top"] or _tb["bottom"]:
        queue_top_bottom_ts["p_master_batcher"] = {
            **_tb,
            "rank": "peak",
            "sample_window_s": 5,
        }

# 引擎侧 running / waiting（mock_per_engine_timeseries.json.gz，mock 引擎
# 自身上报的 per-engine gauge，1s 采样）。
for _side, _role_tag in (("p", "prefill"), ("d", "decode")):
    for _metric in ("running", "waiting"):
        _series = (
            _ts_role_ip_split(mock_per_engine_ts, "mock_engine_" + _metric).get(
                _role_tag
            )
            or {}
        )
        _tb = _top_bottom_rows(_series)
        if _tb["top"] or _tb["bottom"]:
            queue_top_bottom_ts[_side + "_" + _metric] = {
                **_tb,
                "rank": "peak",
                "sample_window_s": 5,
            }

# ---- mock 自报生产口径 TPS 集群级时序（mock_tps_ts，20260901）----
# rtp_llm_* 系列（生产同名指标，纯完成事件记账；/metrics scrape 先
# drain 再快照 → 窗口 = scrape 间隔，G1 轮询 1s → 值即 tokens/s）经
# G1 白名单进 per-engine 时序文件。这里按指标名跨引擎同时间戳求和成
# 集群级行序列：一次 scrape 对全部引擎统一 drain，三指标同 ts；role
# 不拆（prefill 桶只含 context 对、decode 桶只含 generate，跨引擎
# 合并即集群口径）。旧 run（白名单未含该系列）→ 空行 → 报告层省略。
_TPS_COL_NAMES = {
    "rtp_llm_context_tps": "context_tps",
    "rtp_llm_context_tps_with_cache": "context_tps_with_cache",
    "rtp_llm_generate_tps": "generate_tps",
}
_tps_by_ts = defaultdict(dict)
for _tps_base, _tps_col in _TPS_COL_NAMES.items():
    for _role_engines in _ts_role_ip_split(mock_per_engine_ts, _tps_base).values():
        for _engine_pts in _role_engines.values():
            for _ts, _v in _engine_pts:
                _tps_by_ts[_ts][_tps_col] = _tps_by_ts[_ts].get(_tps_col, 0.0) + _v
mock_tps_ts = [{"t": t, **vals} for t, vals in rel_axis(sorted(_tps_by_ts.items()))]

# ---- 引擎侧 KV v2 块池时序（kv_blocks_ts_by_role，20260902）----
# mock /metrics 新块池系列（三态 gauge + 准入/复用/淘汰累计 counter，
# 1s 采样）经 G1 白名单进 per-engine 时序文件。按 role 拆分、同 ts
# 跨引擎求和（块数与累计 counter 均可加）—— aggregate 保留集群和原
# 始语义，报告层除以引擎数（三级回退链）得每引擎平均、对 counter 列
# 累计差分得速率。列语义：
#   total/available/held/referenced_blocks —— 三态块分解（gauge，
#     available = free + 纯 LRU；held = 运行中裸块；referenced =
#     在途请求引用的 key 块）；
#   cache_evictions —— LRU 淘汰累计（容量 + 强制 /cache_evict）；
#   kv_admission_fails —— decode 降级/增长失速累计（prefill 同步拒绝
#     不在此列，另计 lack_mem_rejects）；
#   lack_mem_rejects —— prefill 同步 602 拒绝累计（生产 MALLOC_FAILED
#     同码，enqueue ack 直接拒）；
#   decode_reuse_blocks —— decode 接手自身 LRU 重算命中的复用块累计
#     （KV v2 fix #5 净需求折减量，“越用省越多”正反馈的直接读数）。
# 旧 run（白名单未含该系列）→ 空表 → 报告层整节静默省略。
_KV_BLOCK_COLUMNS = {
    "mock_engine_cache_blocks": "total_blocks",
    "mock_engine_available_blocks": "available_blocks",
    "mock_engine_held_blocks": "held_blocks",
    "mock_engine_referenced_blocks": "referenced_blocks",
    "mock_engine_cache_evictions_total": "cache_evictions",
    "mock_engine_kv_admission_fails_total": "kv_admission_fails",
    "mock_engine_lack_mem_rejects_total": "lack_mem_rejects",
    "mock_engine_decode_reuse_blocks_total": "decode_reuse_blocks",
}
_kv_by_role_ts = defaultdict(lambda: defaultdict(dict))
for _kv_base, _kv_col in _KV_BLOCK_COLUMNS.items():
    for _role, _engines in _ts_role_ip_split(mock_per_engine_ts, _kv_base).items():
        for _pts in _engines.values():
            for _ts, _v in _pts:
                _row = _kv_by_role_ts[_role.lower()][_ts]
                _row[_kv_col] = _row.get(_kv_col, 0.0) + _v
kv_blocks_ts_by_role = {
    _role: [{"t": t, **vals} for t, vals in rel_axis(sorted(ts_map.items()))]
    for _role, ts_map in _kv_by_role_ts.items()
}

# ---- cache 命中率三口径（cache_hit_ts / cache_hit_summary，20260902）----
# 三口径对齐生产观测（报告层「cache 命中率」小节消费）：
#   口径 1｜master 路由（master 以为能复用多少）：master 选引擎时
#     GlobalCacheIndex 前缀匹配的 counter 对
#     flexlb_app_cache_routing_selected_match_{hit,total}_tokens_total
#     （G3 whale-lb app.cache 族，实测只有 role="PREFILL" 变体，sum
#     折叠即集群值）。时序 = counter 相邻同拍正差分得窗口 hit/total
#     增量对 → 窗口命中率；run 级 = 末拍累计比（对齐生产 app.cache
#     族口径，master 代码零改动）。
#   口径 2｜engine key 级（理论）：新 counter mock_engine_cache_key_
#     hits/keys_requested_total（引擎 prefill 准入 prefixHitBlocks 记账
#     点累计，对齐生产 recent_cache_key_hit_count/total_count 口径）
#     跨引擎同拍求和后同法差分。空 bh 请求 0/0 自然不贡献。只有
#     prefill 引擎跑准入记账，但两 counter 同记账点，role 不拆。
#   口径 3｜engine token 级（实际）：ΣhitTokens/Σil（对齐生产 reuse/
#     input）。时序从 mock_tps_ts 逐窗导出 (context_tps_with_cache −
#     context_tps)/context_tps_with_cache（P 完成事件记账，两列同窗
#     口径自洽）；run 级 = cache_saved_tokens（Σ引擎 hit_tokens_total）
#     ÷ ok_input_tokens（client ok 行 Σil，完成请求口径）。
# 差值语义（报告层标注）：master_routing − engine_token = 调度损耗
# （master 匹配到却未复用：路由到非持有引擎/affinity 未采纳/执行窗口
# 内 LRU 淘汰）；engine_key − engine_token = 命中深度覆盖（部分前缀
# 命中：命中 key 但前缀在第 N 块断掉，复用 tokens 不足整 key 数）。
# 旧 run 任一口径数据缺失 → 该列整体缺省（不造 0），summary 相应键
# 缺省，报告层按口径独立省略。


def _counter_pair_ratio(num_pts, den_pts):
    """两条同源累计 counter → (窗口比率时序, run 级末值比).

    相邻同拍样本正差分（counter 重置/回退的负差分窗丢弃）得窗口增量
    对，den 增量 >0 才产出窗口比率；run 级 = 末拍 num/den（counter
    只增，末拍比即末拍前全程累计比）。num/den 按共同 ts 对齐（同源
    系列一次采样同拍输出）。
    """
    num_by_ts = dict(num_pts)
    den_by_ts = dict(den_pts)
    common = sorted(set(num_by_ts) & set(den_by_ts))
    windows = []
    for ts0, ts1 in zip(common, common[1:]):
        num_d = num_by_ts[ts1] - num_by_ts[ts0]
        den_d = den_by_ts[ts1] - den_by_ts[ts0]
        if den_d <= 0 or num_d < 0:
            continue
        windows.append((ts1, num_d / den_d))
    run_ratio = None
    if common and den_by_ts[common[-1]] > 0:
        run_ratio = num_by_ts[common[-1]] / den_by_ts[common[-1]]
    return windows, run_ratio


# 口径 1：master 路由 counter 对（G3 prometheus 差分窗口）。
_route_hit_pts = prom_ts_extract(
    "flexlb_app_cache_routing_selected_match_hit_tokens_total", agg="sum"
)
_route_total_pts = prom_ts_extract(
    "flexlb_app_cache_routing_selected_match_total_tokens_total", agg="sum"
)
_master_route_windows, _master_route_run = _counter_pair_ratio(
    _route_hit_pts, _route_total_pts
)

# 口径 2：engine key 级 counter 对（G1 per-engine，跨引擎同拍求和后
# 差分；prefill 引擎的准入记账，两 counter 同点可比）。
_key_col_map = {
    "mock_engine_cache_key_hits_total": "hits",
    "mock_engine_cache_keys_requested_total": "requested",
}
_key_by_ts = defaultdict(dict)
for _key_base, _key_col in _key_col_map.items():
    for _role_engines in _ts_role_ip_split(mock_per_engine_ts, _key_base).values():
        for _key_pts in _role_engines.values():
            for _ts, _v in _key_pts:
                _krow = _key_by_ts[_ts]
                _krow[_key_col] = _krow.get(_key_col, 0.0) + _v
_engine_key_windows, _engine_key_run = _counter_pair_ratio(
    [(ts, vals["hits"]) for ts, vals in sorted(_key_by_ts.items()) if "hits" in vals],
    [
        (ts, vals["requested"])
        for ts, vals in sorted(_key_by_ts.items())
        if "requested" in vals
    ],
)

# 口径 3：engine token 级 —— mock_tps_ts 逐窗导出（行 t 已是 rel 轴
# 秒，不再 rel_axis；窗口内 with_cache=含复用总吞吐、context=实际计算
# 吞吐，差值即复用节省占比）。
_engine_token_windows = []
for _tps_row in mock_tps_ts:
    _wc = _tps_row.get("context_tps_with_cache")
    _nc = _tps_row.get("context_tps")
    if isinstance(_wc, (int, float)) and isinstance(_nc, (int, float)) and _wc > 0:
        _engine_token_windows.append((_tps_row["t"], max(0.0, _wc - _nc) / _wc))
_engine_token_run = None
if cache_saved_tokens_calc and ok_input_tokens:
    _engine_token_run = cache_saved_tokens_calc / ok_input_tokens

# 三口径按 t（秒，rel 轴）合并成行序列；任一口径缺数据 → 该列整体
# 缺省（旧 run 优雅降级，报告层按列独立画线/省略）。
_cache_hit_by_t = defaultdict(dict)
for _t, _v in rel_axis(_master_route_windows):
    _cache_hit_by_t[round(_t, 1)]["master_routing"] = round(_v, 4)
for _t, _v in rel_axis(_engine_key_windows):
    _cache_hit_by_t[round(_t, 1)]["engine_key"] = round(_v, 4)
for _t, _v in _engine_token_windows:
    _cache_hit_by_t[round(_t, 1)]["engine_token"] = round(_v, 4)
cache_hit_ts = [{"t": t, **vals} for t, vals in sorted(_cache_hit_by_t.items())]

# run 级三口径汇总（各口径独立缺省；counter 末拍读数同时落 summary
# 供报告 KPI 读数行与对账取证）。
cache_hit_summary = {}
if _master_route_run is not None:
    cache_hit_summary["master_routing_hit_pct"] = round(_master_route_run * 100.0, 1)
    if _route_hit_pts and _route_total_pts:
        cache_hit_summary["master_routing_hit_tokens"] = int(
            round(_route_hit_pts[-1][1])
        )
        cache_hit_summary["master_routing_total_tokens"] = int(
            round(_route_total_pts[-1][1])
        )
if _engine_key_run is not None:
    cache_hit_summary["engine_key_hit_pct"] = round(_engine_key_run * 100.0, 1)
    if _key_by_ts:
        _last_key_ts = max(_key_by_ts)
        _last_key_vals = _key_by_ts[_last_key_ts]
        if "hits" in _last_key_vals:
            cache_hit_summary["engine_key_hits"] = int(round(_last_key_vals["hits"]))
        if "requested" in _last_key_vals:
            cache_hit_summary["engine_keys_requested"] = int(
                round(_last_key_vals["requested"])
            )
if _engine_token_run is not None:
    cache_hit_summary["engine_token_hit_pct"] = round(_engine_token_run * 100.0, 1)
    cache_hit_summary["engine_hit_tokens"] = cache_saved_tokens_calc
    cache_hit_summary["engine_input_tokens"] = ok_input_tokens

# ---- token 对账（20260901 纠偏；同日二次修复补在途项）：client 完成 ----
# ---- token vs mock 自报累计 ----
# 原 canvas 2.3 节 input/output 侧 client 对账面板移除后，丢请求 /
# 自报造假的检测能力保留为 validity 项 token_reconciliation_ok
# （fail-closed 断言，与 leak KPI 同族）。数据源与口径：
#   * client 侧 = ok 行 Σil/Σol（ok_input_tokens / ok_output_tokens，
#     完成请求口径，与 mock 完成事件记账语义对齐——被拒/取消行两边
#     都不进分子）。
#   * mock 侧 = Σ mock_tps_ts 窗口值 + 在途项 in-flight Σ。/metrics
#     的 scrape-先-drain 语义保证每个 token 恰好被 drain 一次（漏拍
#     时下窗并入，Σ 对 scrape 抖动稳健），Σ = 截至最后一次 scrape 的
#     累计完成 token。
#   * 在途项 in-flight（20260901 run 20260901_200108 取证）：fire-and-
#     forget（FETCH_OUTPUT_STREAM=0，标准 run）下 client 在 master
#     schedule 成功即记 ok（output_len 记期望值），run 结束仍在
#     decode 引擎在途的长 ol 请求（实测 904 个、Σol≈7.1M、占 client
#     Σ 15.3%）不进 mock Σ（decode 未完成 → 无完成事件可记账）。修复
#     方案：复用 engine_exec/full_e2e 的引擎终态 rid 集合
#     （decode_done_map / prefill_done_map，engine_events.jsonl 的
#     decode_done / prefill_done 行解析），ok 行中 rid 不在对应 done 集的
#     行即在途，其 token 对称补进 mock 侧（input 侧 join prefill done 集、
#     output 侧 join decode done 集；join 判定只用 rid membership 不做
#     时间戳过滤——引擎已记终态即 token 已入 mock 完成事件流，时间戳
#     异常只影响延迟样本有效性不影响记账归属）。rid 缺失/不可解析的
#     ok 行无法证明完成，保守计为在途（fail-closed 方向：宁可放大残差
#     也不掩盖缺口）。
# 容限（input/output 两侧各自）：|client − (mock + in-flight)| ≤
#   max(5% × client, 5 × 每秒峰值 token)。
#   * 5% 相对项：吸收 scrape 窗口边缘 / 时钟对齐残差与取消请求的
#     单侧记账不对称（prefill 已完成但 client 记 error）；健康 run
#     实测完成口径差 ~1%，5% 有 5 倍余量。
#   * 5 × 峰值绝对项：覆盖 G1 scrape 停止后的 drain 尾巴（最后一次
#     scrape 之后、引擎已完成但未被 drain 记账的量，实测残差 1.1M <
#     容限 2.5M）与首尾窗口错位，按 ≤5s × 峰值速率的上界估计。若某
#     run 的 G1 尾巴特别长导致仍 False，属真实采集缺口，如实失败是
#     正确行为。
# 双向检测：丢请求（mock < client）与自报虚高（mock > client）任一
# 超容限即 False。缺数据（mock_tps_ts 空 / 侧系列全零 / client 侧无
# token）→ None（不误报，与现有六项缺输入语义一致；test_valid =
# all 保守判 invalid）。两侧子对账任一可评估即以可评估侧的 AND
# 定值；均不可评估 → None。
# 退化语义：engine_events.jsonl 已 fail-closed（缺失即硬错），终态
# done 集恒可用；全 cancelled 的极端情形 done 集为空且 client 有大量
# 未完成行 → 正常计算（在途 = 全部未完成行）。两侧数值依据输出进
# summary.token_reconciliation 诊断键（validity 不进 canvas 版面，报
# 告层不消费，供取证与排查）。
_mock_in_total = sum(
    r.get("context_tps_with_cache") or 0 for r in mock_tps_ts if isinstance(r, dict)
)
_mock_out_total = sum(
    r.get("generate_tps") or 0 for r in mock_tps_ts if isinstance(r, dict)
)
# 在途项：ok 行 rid 不在引擎终态 done 集的 Σil/Σol（两侧对称，见上
# 块注释）。engine_events.jsonl fail-closed 后终态流必在，done 集恒
# 可用（旧「无引擎终态日志→在途恒 0」的退化分支已随数据层消亡）。
_engine_rids_available = True
inflight_input_tokens = 0
inflight_output_tokens = 0
if _engine_rids_available:
    for d in rows:
        if not is_ok(d):
            continue
        _rid = d.get("request_id")
        try:
            _rid = int(_rid)
        except (TypeError, ValueError):
            _rid = None
        if _rid not in prefill_done_map:
            inflight_input_tokens += d.get("input_len") or 0
        if _rid not in decode_done_map:
            inflight_output_tokens += d.get("output_len") or 0
token_recon_diag = None
if isinstance(validity_checks_calc, dict):
    _token_recon_subs = []
    token_recon_diag = {
        "engine_rids_available": _engine_rids_available,
        "input": None,
        "output": None,
    }
    for _side, _client_tok, _mock_tok, _infl_tok, _peaks in (
        (
            "input",
            ok_input_tokens,
            _mock_in_total,
            inflight_input_tokens,
            [(p.get("input_tokens") or 0) for p in per_second]
            + [r.get("context_tps_with_cache") or 0 for r in mock_tps_ts],
        ),
        (
            "output",
            ok_output_tokens,
            _mock_out_total,
            inflight_output_tokens,
            [(p.get("output_tokens_completed") or 0) for p in per_second]
            + [r.get("generate_tps") or 0 for r in mock_tps_ts],
        ),
    ):
        if _client_tok <= 0 or _mock_tok <= 0:
            continue
        _peak = max(_peaks) if _peaks else 0
        _tol = max(0.05 * _client_tok, 5 * _peak)
        _residual = abs(_client_tok - (_mock_tok + _infl_tok))
        _token_recon_subs.append(_residual <= _tol)
        token_recon_diag[_side] = {
            "client_tokens": _client_tok,
            "mock_tokens": round(_mock_tok, 1),
            "inflight_tokens": _infl_tok,
            "residual": round(_residual, 1),
            "tolerance": round(_tol, 1),
            "pass": _residual <= _tol,
        }
    validity_checks_calc["token_reconciliation_ok"] = (
        all(_token_recon_subs) if _token_recon_subs else None
    )
    test_valid_calc = all(v is True for v in validity_checks_calc.values())

# Per-dispatch batch size gauge (engine.balancing.master.batch.size, tags
# role + engineIp + reason, reported once per dispatch). Per reason the
# per-engine values are averaged: each engine's gauge holds the size of
# its most recent dispatch with that reason, so the engine average is the
# cluster-representative batch size for that reason at sample time.
DISPATCH_BATCH_SIZE_BASE = "flexlb_app_engine_balancing_master_batch_size"
batch_size_series = prom_ts_extract_labeled(
    DISPATCH_BATCH_SIZE_BASE, "reason", agg="avg"
)
dispatch_batch_size_ts = []
if batch_size_series:
    bs_by_ts = defaultdict(dict)
    for reason, pts in batch_size_series.items():
        for ts, v in pts:
            bs_by_ts[ts][reason] = round(v, 2)
    dispatch_batch_size_ts = [
        {"t": t, **vals} for t, vals in rel_axis(sorted(bs_by_ts.items()))
    ]

# Terminal batch size distribution from the end-of-run prometheus_after
# snapshot: each engine's batch_size gauge holds the size of its most recent
# dispatch with that reason, so the snapshot yields a per-reason distribution
# across engines (the 1s timeline whitelist only picked this gauge up after
# the poller fix — the final distribution is available on every run).
master_prom_after = master_json.get("prometheus_after") or {}
batch_size_final = {}
if isinstance(master_prom_after, dict):
    bs_reason_re = re.compile(r'reason="([^"]*)"')
    bs_final_vals = defaultdict(list)
    for k, v in master_prom_after.items():
        if not isinstance(v, (int, float)):
            continue
        key = str(k)
        if key.split("{", 1)[0] != DISPATCH_BATCH_SIZE_BASE:
            continue
        m = bs_reason_re.search(key)
        if not m:
            continue
        bs_final_vals[m.group(1)].append(float(v))
    for reason, vs in bs_final_vals.items():
        vs.sort()
        n = len(vs)
        batch_size_final[reason] = {
            "engines": n,
            "min": vs[0],
            "p50": vs[n // 2],
            "p90": vs[min(n - 1, int(n * 0.9))],
            "max": vs[-1],
            "avg": round(sum(vs) / n, 2),
        }

# G3 dispatch reason counters -> per-second dispatch rate per reason.
# dispatch_reason_total{reason=...} is a monotonically increasing counter
# sampled once per second by the master prometheus poller; the positive
# delta between consecutive samples divided by the sample gap is the
# per-second dispatch rate of that decision reason. A counter reset
# (negative delta) drops that interval instead of fabricating a spike.
DISPATCH_REASON_BASE = "flexlb_app_engine_balancing_master_dispatch_reason_total"
reason_series = prom_ts_extract_labeled(DISPATCH_REASON_BASE, "reason")
reason_rate_rows = []
if reason_series:
    rate_by_ts = defaultdict(dict)
    for reason, pts in reason_series.items():
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            dt_s = (t1 - t0) / 1000.0
            if dt_s <= 0 or v1 < v0:
                continue
            rate_by_ts[t1][reason] = round((v1 - v0) / dt_s, 2)
    reason_rate_rows = sorted(rate_by_ts.items())
dispatch_reason_ts = [{"t": t, **vals} for t, vals in rel_axis(reason_rate_rows)]

# consolidate integrity markers (consolidate_run_outputs.py): how the
# final_snapshot was obtained (live HTTP fetch vs stale fallback) and
# whether the slo analysis predates this run's client_events data. Empty for
# pre-integrity consolidations; the generator then stays silent.
integrity = {}
if per_second_unstamped:
    # Degradation marker: rows that carry no usable send timestamp (0/None)
    # are excluded from per_second; report readers must not expect
    # sum(per_second.arrivals) == total_requests.
    integrity["per_second_rows_without_send_ts"] = per_second_unstamped
# Degradation marker (full_e2e / birth-axis decode exec): scheduled-ok rows
# with no normal engine-side decode terminal row to join against (in-flight
# at run end / cancelled decode / truncated engine_events stream). Not
# fabricated into either metric.
if full_e2e_join_miss:
    integrity["full_e2e_join_miss"] = full_e2e_join_miss
# Degradation marker (birth-axis prefill exec): scheduled-ok rows with no
# normal engine-side prefill batch-completion row to join against. Not
# fabricated into the metric. Only fires when the join runs (a stream whose
# terminal rows are all cancelled leaves the map empty → no join attempted,
# no miss counted).
if prefill_exec_join_miss:
    integrity["prefill_exec_join_miss"] = prefill_exec_join_miss
# cancel 按角色拆分的降级标记：cancelled_rids 里无法在 master 终态行
# 定位时刻的 rid 数（这些 tracked cancel 事件被丢弃、不计入
# prefill/decode cancel 线，不编造时刻）。
if any(_cancel_role_unmatched.values()):
    integrity["cancel_role_events_without_terminal_ts"] = {
        k: v for k, v in _cancel_role_unmatched.items() if v
    }
if isinstance(mock_payload, dict) and mock_payload.get("final_snapshot_source"):
    integrity["final_snapshot_source"] = mock_payload["final_snapshot_source"]
if isinstance(master_json, dict) and master_json.get("slo_integrity"):
    integrity["slo_integrity"] = master_json["slo_integrity"]

_run_params = run_meta.get("params") or {}
if "fetch_output_stream" in _run_params:
    _fos_raw = str(_run_params["fetch_output_stream"]).strip()
    fetch_output_stream = _fos_raw not in ("0", "false", "False", "no")
else:
    # legacy runs recorded the inverted switch as schedule_only
    _legacy = str(_run_params.get("schedule_only", "0")).strip()
    fetch_output_stream = _legacy not in ("1", "true", "True")

# ---- meta 补充：报告头部三层取数源（20260902 条件/结果/详情重构）----
# subtitle 实验条件（send_mode / replay_speed / 名义 QPS / ramp / duration）
# 与 detail 层（数据集 / 配置 / 代码版本）从 run_meta.params 提升进
# aggregate meta；canvas_report_gen.py 优先读这里，旧 aggregate 无键时
# 回退同目录 run_meta.json。值缺失（None / 空串）不写键——fail-closed
# 留空，不硬错。远端 rsync 树无 .git：git_branch / git_commit 由调用方
# （重聚合命令）经 FLEXLB_GIT_BRANCH / FLEXLB_GIT_COMMIT 注入。


def _meta_int(raw):
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


_meta_conditions = {}
for _k in (
    "send_mode",
    "trace_file",
    "trace_file_sha256",
    "flexlb_config",
    "performance_file",
    "process_config_file",
):
    _v = str(_run_params.get(_k) or "").strip()
    if _v:
        _meta_conditions[_k] = _v
for _k in (
    "replay_speed",
    "send_mode_qps",
    "ramp_up_seconds",
    "duration_s",
    "trace_file_lines",
):
    _v = _meta_int(_run_params.get(_k))
    if _v is not None:
        _meta_conditions[_k] = _v
for _k, _env in (
    ("git_branch", "FLEXLB_GIT_BRANCH"),
    ("git_commit", "FLEXLB_GIT_COMMIT"),
):
    _v = str(os.environ.get(_env) or "").strip()
    if _v:
        _meta_conditions[_k] = _v

out = {
    "meta": {
        "run_dir": os.path.basename(run_dir),
        "fetch_output_stream": fetch_output_stream,
        **_meta_conditions,
    },
    "summary": {
        # ---- rows 唯一指标源（no-backward-compat）：全部自算，rows 缺失
        # ---- 即 None；不透传 summary 键，不做旧键名映射。 ----
        "total_requests": len(rows) if _have_rows else None,
        "success_count": _success_count_calc if _have_rows else None,
        "error_count": _error_count_calc if _have_rows else None,
        "error_rate": error_rate_calc,
        # 错误构成（具名子桶细分，含无时间戳行，与 error_count 同口径；
        # rows 缺失时空 dict（无数据，非透传）。
        "error_breakdown": dict(error_breakdown),
        "sent_task_count": _sent_task_count if _have_rows else None,
        "load_client_workers": _load_client_workers,
        "actual_rpc_start_count": _actual_rpc_start_count if _have_rows else None,
        "recorded_result_count": _recorded_result_count if _have_rows else None,
        "completed_count": completed_count if _have_rows else None,
        "elapsed_s": elapsed_s_calc,
        "actual_send_qps": actual_send_qps_calc,
        "success_qps": success_qps_calc,
        "error_qps": error_qps_calc,
        "completed_qps": completed_qps_calc,
        # client 侧 token 吞吐（完成请求口径，20260901）：ok 行 Σil/Σol
        # ÷ elapsed；与 mock 自报 rtp_llm_* 生产口径对表（mock 侧为
        # 完成事件记账的 1s 窗口值，client 侧为全程平均）。
        "input_token_tps": input_token_tps_calc,
        "output_token_tps": output_token_tps_calc,
        # KV 复用收益量化（20260901）：引擎累计 Σhit_tokens_total
        # （final_snapshot，consolidate live fetch 链路）。
        "cache_saved_tokens": cache_saved_tokens_calc,
        # cache 命中率 run 级三口径汇总（20260902）：master 路由 /
        # engine key 级（理论）/ engine token 级（实际），各口径独立
        # 缺省（旧 run 无新 counter/系列 → 键缺省，不造 0）。报告层
        # 「cache 命中率」小节 KPI 读数行消费（caption 注明与生产
        # app.cache 族 / recent_cache_key_hit / reuse-input 的对齐
        # 关系）。
        "cache_hit_summary": cache_hit_summary,
        # token 对账诊断（20260901 in-flight 修复）：两侧 client/mock/
        # in-flight/residual/tolerance 取证值（validity 判定的数值依据，
        # 报告层不消费；rows 缺失 → None）。
        "token_reconciliation": token_recon_diag,
        "client_pacing_lag_ms": _pacing_dist if _have_rows else None,
        # server_* 直读：server_latency 是当前格式正式输入，非 rows 依赖。
        "server_arrival_qps": server_latency.get("arrival_qps"),
        "schedule_latency_source": _schedule_source_calc,
        "schedule_latency_ms": _schedule_latency_calc,
        # 跨两侧全链路分位（聚合层新算，非 summary.json 原生字段）；
        # full_e2e = client 发出 → 引擎 decode 正常终态，按 request_id
        # 关联，schedule-only（FETCH=0）下也覆盖完整链路。
        "full_e2e_latency_ms": full_e2e_latency_ms,
        "server_stage_latency_ms": _server_stage_calc,
        # ttft/e2e 全程分位（聚合层自算，幸存者口径）；rows 缺失即 None。
        "ttft_latency_ms": _ttft_summary_calc if _have_rows else None,
        "e2e_latency_ms": _e2e_summary_calc if _have_rows else None,
        "validity_checks": validity_checks_calc,
        "test_valid": test_valid_calc,
    },
    "batch": {
        "mock_last": slo.get("mock", {}).get("last"),
    },
    "per_second": per_second,
    "master_arrivals_ts": master_arrivals_ts,
    "queue_timeseries": queue_ts,
    "engine_dist": engine_dist,
    "stage_latency_ts": stage_latency_ts,
    "engine_exec_ts": engine_exec_ts,
    "cancel_qps_ts": cancel_qps_ts,
    "integrity": integrity,
    "process_ts": process_ts,
    "inflight_ts": inflight_ts,
    "inflight_age_ts": inflight_age_ts,
    "inflight_age_by_role": inflight_age_by_role,
    "kv_ts": kv_ts,
    "batcher_ts": batcher_ts,
    "batcher_ts_by_role": batcher_ts_by_role,
    "batcher_top_engines_ts": batcher_top_engines_ts,
    "queue_top_bottom_ts": queue_top_bottom_ts,
    # mock 自报生产口径 TPS 集群级时序（rtp_llm_* 跨引擎求和，1s 窗口
    # 完成事件记账；报告层 2.3 节对账图消费）。
    "mock_tps_ts": mock_tps_ts,
    # 引擎侧 KV v2 块池时序（三态 gauge + 准入/复用/淘汰 counter，按
    # role 拆分跨引擎求和；报告层 5. KV 块池面板消费——除以引擎数得
    # 每引擎平均、counter 列累计差分得速率）。
    "kv_blocks_ts_by_role": kv_blocks_ts_by_role,
    # cache 命中率三口径时序（master_routing/engine_key/engine_token
    # 列按窗口命中率，各口径独立缺省；报告层 5c「cache 命中率」小节
    # 消费——「master 路由 vs engine 执行」双曲线（差值=调度损耗）+
    # 「key 级理论 vs token 级实际」双曲线（差值=命中深度覆盖））。
    "cache_hit_ts": cache_hit_ts,
    "dispatch_reason_ts": dispatch_reason_ts,
    "dispatch_batch_size_ts": dispatch_batch_size_ts,
    "batch_size_final": batch_size_final,
}
json.dump(out, sys.stdout)
