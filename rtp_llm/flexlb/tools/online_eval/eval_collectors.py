#!/usr/bin/env python3
"""Background collectors for run_online_eval.sh: threads of one
stdlib-only process, one thread per enabled poller lane.

run_online_eval.sh starts one process per run and passes only the lanes
whose guards are on —
                      G1 mock per-engine prometheus (mock control port),
                      G3 master business prometheus (management port),
                      G5 process CPU/RSS sampling.

(G6 counter poller and G4 inflight poller were collapsed into G3: their
consumed series now ride the G3 whitelist — see MASTER_PROMETHEUS_PREFIXES
— and aggregate_canvas_run.py derives master_arrivals_ts / inflight_ts
from the prometheus timeline.)

Each thread is best-effort: a failed round is skipped, never fatal, and
each round sleeps the remainder of the poll interval. The output files
are append-mode one-shot series consumed by consolidate_run_outputs.py /
aggregate_canvas_run.py — byte-format compatibility is a hard requirement.

Configuration arrives via argv, shell-expanded by run_online_eval.sh (which
stays the single source of truth for env defaults). SIGTERM/SIGINT set a
stop event: the in-flight round finishes, buffered output is flushed, then
the process exits.
"""

import argparse
import collections
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
    # TTL-eviction two-level counter (scheduler request-slot ledger sweep
    # + P/D endpoint-ledger orphan sweeps).  Sparse: the master reports a
    # series only once it is non-zero, so rounds with no lines for this
    # prefix mean "no eviction in that window", not a collection gap; the
    # series persists forever after first appearance (cumulative counter).
    # Stress-run observability side benefit — the case-side assertions
    # scrape the endpoint directly (EngineOps.master_ttl_eviction_counts),
    # not this G3 file.
    "flexlb_app_flexlb_inflight_ttl",
    "flexlb_app_engine_balancing_master_dispatch_reason_total",
    "flexlb_app_engine_balancing_master_batch_size",
    # G6/G4 collapse — the G3 lane is now the sole master-plane collector,
    # so the series those retired pollers fed to aggregate moved here:
    # arrival/completion counters (request_count_total{priority}, summed
    # + differenced by aggregate_canvas_run into master_arrivals_ts;
    # all_qps_total{code} the completion side) and the five inflight
    # gauges (scheduler direct, per-engine prefill batch/request counts,
    # per-endpoint decode reserved/running — rebuilt into inflight_ts).
    # Keep in sync with FLEXLB_MONITOR_METRIC_WHITELIST in
    # run_online_eval.sh (the master-side trim at the source).
    "flexlb_auto_tpm_request_count",
    "flexlb_app_engine_balancing_master_all_qps",
    "flexlb_app_flexlb_scheduler_inflight_size",
    "flexlb_app_flexlb_inflight_batch_count",
    "flexlb_app_flexlb_inflight_request_count",
    "flexlb_auto_tpm_decode_reserved_count",
    "flexlb_auto_tpm_decode_running_count",
)

_STOP = threading.Event()
# After a stop request, wait at most this long for the in-flight round
# (bounded by the 2s urlopen/ps timeout) before exiting anyway.
_SHUTDOWN_GRACE_S = 3.0


def _request_stop(signum, frame):
    # Re-entry guard: Event.set() acquires the Event's underlying Condition
    # lock, which is NOT reentrant — a second signal delivered while the
    # handler sits inside that critical section re-enters the handler and
    # self-deadlocks the whole process (observed as one leaked frozen
    # collector per run when the shell stop loop sent 5 rapid SIGTERMs to
    # the unified process; py-spy showed 3-level recursive handler frames).
    # Mask BOTH stop signals on entry so neither a same-signal burst nor a
    # SIGTERM/SIGINT cross can nest — the process is on its way out, so
    # ignoring further stop signals is the intended semantics. The
    # try/except is a last-resort belt: even if set() were ever to raise,
    # the handler must return, never hang.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _STOP.set()
    except Exception:
        pass


def _sleep_remaining(started, interval_s):
    """Interval-remainder sleep, identical to the heredocs'
    time.sleep(max(0.0, interval_s - (time.time() - started))), except that
    a requested stop interrupts the wait so shutdown is prompt."""
    remaining = interval_s - (time.time() - started)
    if remaining > 0:
        _STOP.wait(remaining)


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


# Poller lane table — one entry per collector thread. A lane is enabled by
# passing its (enable, out) argv pair together. A lane with an interval flag
# controls its own cadence: a None default falls back to the shared
# --secondary-interval.
_Lane = collections.namedtuple(
    "_Lane",
    "enable_flag out_flag interval_flag interval_default target name desc",
)

_LANES = (
    _Lane(
        "--mock-port",
        "--mock-out",
        "--mock-interval",
        None,
        run_mock_per_engine_poller,
        "mock-per-engine-poller",
        "G1 mock per-engine prometheus (mock control port)",
    ),
    _Lane(
        "--prometheus-port",
        "--prometheus-out",
        None,
        None,
        run_master_prometheus_poller,
        "master-prometheus-poller",
        "G3 master business prometheus (management port)",
    ),
    _Lane(
        "--pid-file",
        "--process-out",
        None,
        None,
        run_process_usage_poller,
        "process-usage-poller",
        "G5 process CPU/RSS sampling",
    ),
)


def _argv_name(flag):
    """--prometheus-port -> the args attribute name prometheus_port."""
    return flag.lstrip("-").replace("-", "_")


def main():
    parser = argparse.ArgumentParser(
        description="run_online_eval.sh secondary collectors: one process, "
        "one thread per enabled poller lane (G1/G3/G5)."
    )
    parser.add_argument(
        "--secondary-interval",
        type=float,
        default=1.0,
        help="SECONDARY_POLL_INTERVAL_S — shared poll interval for lanes "
        "without their own (default: 1)",
    )
    for lane in _LANES:
        parser.add_argument(lane.enable_flag, help=lane.desc)
        parser.add_argument(lane.out_flag, help=f"{lane.desc} — output file")
        if lane.interval_flag is not None:
            fallback = (
                "shared --secondary-interval"
                if lane.interval_default is None
                else lane.interval_default
            )
            parser.add_argument(
                lane.interval_flag,
                type=float,
                default=lane.interval_default,
                help=f"{lane.desc} — poll interval (default: {fallback})",
            )
    args = parser.parse_args()

    threads = []
    for lane in _LANES:
        enable_value = getattr(args, _argv_name(lane.enable_flag))
        out_value = getattr(args, _argv_name(lane.out_flag))
        if bool(enable_value) != bool(out_value):
            parser.error(
                f"{lane.enable_flag} and {lane.out_flag} must be passed together"
            )
        if not enable_value:
            continue
        interval = args.secondary_interval
        if lane.interval_flag is not None:
            lane_interval = getattr(args, _argv_name(lane.interval_flag))
            if lane_interval is not None:
                interval = lane_interval
        threads.append(
            threading.Thread(
                target=lane.target,
                args=(enable_value, out_value, interval),
                name=lane.name,
                daemon=True,
            )
        )
    if not threads:
        print(
            "eval_collectors: no collector lane enabled (all START_* guards off)",
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
