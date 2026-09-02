#!/usr/bin/env python3
"""Background collectors extracted from run_online_eval.sh's heredoc pollers.

Stage 2 of the unified-python-entry refactor: the five python3 heredoc
pollers that run_online_eval.sh used to spawn as independent background
processes now live here as threads of one stdlib-only process.

Groups (run_online_eval.sh starts one process per group):
  --group counter   : 1 thread — master arrival/completion counter poller
                      (GET {addr}/rtp_llm/server_latency ->
                       master_counters_timeseries.txt).
  --group secondary : up to 4 threads —
                      G1 mock per-engine prometheus (mock control port),
                      G3 master business prometheus (management port),
                      G4 master inflight snapshot,
                      G5 process CPU/RSS sampling.

Each thread is a line-by-line port of the original heredoc: same URL
construction, same parsing/whitelisting, same output line format, same
best-effort semantics (a failed round is skipped, never fatal), same
interval arithmetic (sleep the remainder of the interval after each round).
The output files are append-mode one-shot series consumed by
consolidate_run_outputs.py / aggregate_canvas_run.py — byte-format
compatibility is a hard requirement.

Configuration arrives via argv, shell-expanded by run_online_eval.sh (which
stays the single source of truth for env defaults), mirroring the old
heredoc argv convention. SIGTERM/SIGINT set a stop event: the in-flight
round finishes, buffered output is flushed, then the process exits.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

# G1 C whitelist — the analyzer-consumed mock per-engine series: the
# running/waiting pair (queue depth curves) plus the production-caliber
# TPS trio (rtp_llm_*, completion-event accounting in 1s scrape windows;
# consumed by aggregate mock_tps_ts and the report-layer 2.3 对账图) and
# the KV v2 block-pool family (three-state block gauges + admission /
# reuse / eviction counters; consumed by aggregate kv_blocks_ts_by_role
# and the report-layer 5. KV 块池面板 — every entry below has a
# downstream consumer, do not add dead keys).
MOCK_KEEP_SERIES = {
    "mock_engine_running",
    "mock_engine_waiting",
    "rtp_llm_context_tps",
    "rtp_llm_context_tps_with_cache",
    "rtp_llm_generate_tps",
    "mock_engine_cache_blocks",
    "mock_engine_available_blocks",
    "mock_engine_held_blocks",
    "mock_engine_referenced_blocks",
    "mock_engine_cache_evictions_total",
    "mock_engine_kv_admission_fails_total",
    "mock_engine_lack_mem_rejects_total",
    "mock_engine_decode_reuse_blocks_total",
    # Key-level cache-hit pair (production recent_cache_key_hit_count /
    # total_count caliber): cumulative counters recorded at the prefill
    # admission hit computation; consumed by aggregate cache_hit_ts /
    # cache_hit_summary and the report-layer cache 命中率面板.
    "mock_engine_cache_key_hits_total",
    "mock_engine_cache_keys_requested_total",
}

# G3 C whitelist — every entry is a consumer-backed series (B3 queue curves,
# M3/S7 inflight age, S7 hit-ratio curves, dispatch reasons).
MASTER_PROMETHEUS_PREFIXES = (
    "flexlb_app_cache_",
    "flexlb_app_flexlb_batcher_queue_size",
    "flexlb_app_flexlb_inflight_max_age_ms",
    "flexlb_app_engine_balancing_master_dispatch_reason_total",
    "flexlb_app_engine_balancing_master_batch_size",
)

_STOP = threading.Event()
# After a stop request, wait at most this long for the in-flight round
# (bounded by the 2s urlopen/ps timeout) before exiting anyway.
_SHUTDOWN_GRACE_S = 3.0


def _request_stop(signum, frame):
    _STOP.set()


def _sleep_remaining(started, interval_s):
    """Interval-remainder sleep, identical to the heredocs'
    time.sleep(max(0.0, interval_s - (time.time() - started))), except that
    a requested stop interrupts the wait so shutdown is prompt."""
    remaining = interval_s - (time.time() - started)
    if remaining > 0:
        _STOP.wait(remaining)


def run_master_counter_poller(http_addr, out_path, interval_s):
    """Counter poller (was: start_master_counter_poller heredoc).

    GET http://{addr}/rtp_llm/server_latency -> cumulative arrival/completion
    counters, one kv line per round."""
    url = f"http://{http_addr}/rtp_llm/server_latency"
    with open(out_path, "a", encoding="utf-8") as out:
        while not _STOP.is_set():
            started = time.time()
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    data = json.load(response)
                out.write(
                    f"ts_epoch_ms={int(started * 1000)} "
                    f"arrival_count={data.get('arrival_count', 0)} "
                    f"completion_count={data.get('completion_count', 0)}\n"
                )
                out.flush()
            except Exception:
                pass  # master briefly unavailable; skip this sample
            _sleep_remaining(started, interval_s)


def run_mock_per_engine_poller(port, out_path, interval_s):
    """G1 (was: start_mock_per_engine_poller heredoc).

    GET http://127.0.0.1:{port}/metrics?per_engine=true, keep only the
    analyzer-consumed series, one "# ts=" grouped block per round."""
    keep = MOCK_KEEP_SERIES
    url = f"http://127.0.0.1:{port}/metrics?per_engine=true"
    with open(out_path, "a", encoding="utf-8") as out:
        while not _STOP.is_set():
            started = time.time()
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    body = response.read().decode("utf-8", "replace")
                # C: keep only the analyzer-consumed series — the raw
                # endpoint still emits the full ~25-per-engine surface (server
                # cost unchanged), but the appended bytes drop to a small
                # fraction of it.
                kept = [
                    line
                    for line in body.splitlines()
                    if not line.startswith("#")
                    and line.split("{", 1)[0].split(" ", 1)[0] in keep
                ]
                if kept:
                    out.write(f"# ts={int(started * 1000)}\n")
                    out.write("\n".join(kept) + "\n")
                    out.flush()
            except Exception:
                pass  # control plane briefly unavailable; skip this sample
            _sleep_remaining(started, interval_s)


def run_master_prometheus_poller(port, out_path, interval_s):
    """G3 (was: start_master_prometheus_poller heredoc).

    GET management-port actuator/prometheus (fallback /prometheus), keep
    only whitelisted prefixes, one "# ts=" grouped block per round."""
    urls = [
        f"http://127.0.0.1:{port}/{path}"
        for path in ("actuator/prometheus", "prometheus")
    ]
    prefixes = MASTER_PROMETHEUS_PREFIXES
    with open(out_path, "a", encoding="utf-8") as out:
        while not _STOP.is_set():
            started = time.time()
            for url in urls:
                try:
                    with urllib.request.urlopen(url, timeout=2) as response:
                        body = response.read().decode("utf-8", "replace")
                except Exception:
                    continue  # try the next path / skip this sample
                kept = [line for line in body.splitlines() if line.startswith(prefixes)]
                if kept:
                    out.write(f"# ts={int(started * 1000)}\n")
                    out.write("\n".join(kept) + "\n")
                    out.flush()
                break
            _sleep_remaining(started, interval_s)


def run_master_inflight_poller(http_addr, out_path, interval_s):
    """G4 (was: start_master_inflight_poller heredoc).

    GET http://{addr}/rtp_llm/inflight_status, one JSONL line per round."""
    url = f"http://{http_addr}/rtp_llm/inflight_status"
    with open(out_path, "a", encoding="utf-8") as out:
        while not _STOP.is_set():
            started = time.time()
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    payload = json.load(response)
                out.write(
                    json.dumps(
                        {"ts_epoch_ms": int(started * 1000), "inflight": payload}
                    )
                    + "\n"
                )
                out.flush()
            except Exception:
                pass  # master briefly unavailable; skip this sample
            _sleep_remaining(started, interval_s)


def run_process_usage_poller(pid_file, out_path, interval_s):
    """G5 (was: start_process_usage_poller heredoc).

    ps -o pid,%cpu,rss,etime over the "<pid> <label>" list re-read from
    pid_file every round (load-client workers are appended there after they
    fork); exited pids are tolerated, a wholly dead pidlist skips the
    round."""
    with open(out_path, "a", encoding="utf-8") as out:
        while not _STOP.is_set():
            started = time.time()
            try:
                entries = []
                with open(pid_file, "r", encoding="utf-8") as pids:
                    for line in pids:
                        parts = line.split()
                        if len(parts) == 2 and parts[0].isdigit():
                            entries.append((parts[0], parts[1]))
                if entries:
                    result = subprocess.run(
                        [
                            "ps",
                            "-o",
                            "pid,%cpu,rss,etime",
                            "-p",
                            ",".join(pid for pid, _ in entries),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0:
                        rows = {}
                        for line in result.stdout.splitlines()[1:]:
                            cols = line.split(None, 3)
                            if len(cols) == 4:
                                rows[cols[0]] = cols
                        for pid, label in entries:
                            cols = rows.get(pid)
                            if cols:
                                out.write(
                                    f"ts_epoch_ms={int(started * 1000)} label={label} "
                                    f"pid={pid} cpu_pct={cols[1]} rss_kb={cols[2]} "
                                    f"etime={cols[3]}\n"
                                )
                        out.flush()
            except Exception:
                pass  # best-effort sampling; skip this round
            _sleep_remaining(started, interval_s)


def _supervise(threads):
    """Join the collector threads; after a stop request, force-exit if the
    in-flight round does not finish within the grace window."""
    grace_deadline = None
    while True:
        alive = False
        for thread in threads:
            thread.join(timeout=0.2)
            if thread.is_alive():
                alive = True
        if not alive:
            return
        if _STOP.is_set():
            if grace_deadline is None:
                grace_deadline = time.time() + _SHUTDOWN_GRACE_S
            elif time.time() >= grace_deadline:
                os._exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Background collectors extracted from run_online_eval.sh "
        "heredoc pollers (one process, one thread per poller)."
    )
    parser.add_argument(
        "--group",
        required=True,
        choices=("counter", "secondary"),
        help="counter = master counter poller thread; "
        "secondary = up to 4 collector threads (G1/G3/G4/G5)",
    )
    parser.add_argument(
        "--counter-http-addr",
        help="master HTTP addr (host:port) for the counter poller",
    )
    parser.add_argument(
        "--counter-out", help="master_counters_timeseries.txt output path"
    )
    parser.add_argument(
        "--counter-interval",
        type=float,
        default=1.0,
        help="MASTER_COUNTER_POLL_INTERVAL_S (default: 1)",
    )
    parser.add_argument(
        "--secondary-interval",
        type=float,
        default=1.0,
        help="SECONDARY_POLL_INTERVAL_S for G3/G4/G5 (default: 1)",
    )
    parser.add_argument(
        "--mock-port", help="mock control port (MOCK_BASE_GRPC_PORT-1) for G1"
    )
    parser.add_argument("--mock-out", help="mock_metrics_per_engine.prom output path")
    parser.add_argument(
        "--mock-interval",
        type=float,
        default=None,
        help="MOCK_PER_ENGINE_POLL_INTERVAL_S (default: secondary interval)",
    )
    parser.add_argument("--prometheus-port", help="master management port for G3")
    parser.add_argument(
        "--prometheus-out", help="master_prometheus_timeseries.prom output path"
    )
    parser.add_argument(
        "--inflight-http-addr", help="master HTTP addr (host:port) for G4"
    )
    parser.add_argument(
        "--inflight-out", help="master_inflight_timeseries.jsonl output path"
    )
    parser.add_argument("--pid-file", help="process_poll_pids.txt path for G5")
    parser.add_argument(
        "--process-out", help="process_usage_timeseries.txt output path"
    )
    args = parser.parse_args()

    threads = []
    if args.group == "counter":
        if not args.counter_http_addr or not args.counter_out:
            parser.error(
                "--group counter requires --counter-http-addr and --counter-out"
            )
        threads.append(
            threading.Thread(
                target=run_master_counter_poller,
                args=(args.counter_http_addr, args.counter_out, args.counter_interval),
                name="master-counter-poller",
                daemon=True,
            )
        )
    else:
        if bool(args.mock_port) != bool(args.mock_out):
            parser.error("--mock-port and --mock-out must be passed together")
        if bool(args.prometheus_port) != bool(args.prometheus_out):
            parser.error(
                "--prometheus-port and --prometheus-out must be passed together"
            )
        if bool(args.inflight_http_addr) != bool(args.inflight_out):
            parser.error(
                "--inflight-http-addr and --inflight-out must be passed together"
            )
        if bool(args.pid_file) != bool(args.process_out):
            parser.error("--pid-file and --process-out must be passed together")
        if args.mock_port:
            mock_interval = (
                args.mock_interval
                if args.mock_interval is not None
                else args.secondary_interval
            )
            threads.append(
                threading.Thread(
                    target=run_mock_per_engine_poller,
                    args=(args.mock_port, args.mock_out, mock_interval),
                    name="mock-per-engine-poller",
                    daemon=True,
                )
            )
        if args.prometheus_port:
            threads.append(
                threading.Thread(
                    target=run_master_prometheus_poller,
                    args=(
                        args.prometheus_port,
                        args.prometheus_out,
                        args.secondary_interval,
                    ),
                    name="master-prometheus-poller",
                    daemon=True,
                )
            )
        if args.inflight_http_addr:
            threads.append(
                threading.Thread(
                    target=run_master_inflight_poller,
                    args=(
                        args.inflight_http_addr,
                        args.inflight_out,
                        args.secondary_interval,
                    ),
                    name="master-inflight-poller",
                    daemon=True,
                )
            )
        if args.pid_file:
            threads.append(
                threading.Thread(
                    target=run_process_usage_poller,
                    args=(args.pid_file, args.process_out, args.secondary_interval),
                    name="process-usage-poller",
                    daemon=True,
                )
            )
        if not threads:
            print(
                "eval_collectors: no secondary collector enabled "
                "(all START_* guards off)",
                file=sys.stderr,
            )
            return 0

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    for thread in threads:
        thread.start()
    _supervise(threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
