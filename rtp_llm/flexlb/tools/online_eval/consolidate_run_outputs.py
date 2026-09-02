#!/usr/bin/env python3
"""Consolidate an online_eval run directory into per-component JSON + log pairs.

Target layout (run root), one JSON + one log per component:

  run_meta.json              flexlb_env.txt + client_env.json contents + startup
                             params snapshot + embedded performance/process-config
                             JSON (performance_json / process_config_json, null
                             when the referenced file is missing) + process_usage
                             per-second CPU/RSS timeline
  mock.json                  java_mock_stats timeline + final cluster snapshot
                             + endpoints summary + per_engine_file pointer /
                             per_engine_sample_count (the per-engine prometheus
                             timeline itself is A-split into
                             mock_per_engine_timeseries.json.gz — at 1250
                             engines x 120s the embedded key approached 1GB)
  mock_per_engine_timeseries.json.gz
                             gzip-streamed G1 per-second per-engine prometheus
                             timeline ([{ts, metrics: {name{labels}: value}}]
                             groups); kept in place, referenced from mock.json
  mock.log                   mock_engine.log (verbatim prefix) + gc log appended
  master.json                master counter timeseries + prometheus-after dict
                             + prometheus_timeseries (per-second filtered
                             flexlb_app_*/JVM series) + inflight_timeseries
                             (per-second /rtp_llm/inflight_status JSONL)
                             + master_info before/after + slo batch summary
  master.log                 flexlb_logs/application.log (verbatim prefix) with
                             flexlb.log / sync.log / sync_consistency.log and the
                             run-root flexlb.log (master stdout) appended
  client.json                server_latency.json + slo_batch_analysis.json
                             embedded, plus per_request_source metadata
                             (Phase B: no summary base — the load client
                             records raw rows only and aggregate_canvas_run.py
                             is the single derived-statistics source)
  client.log                 client_shard_*.stdout merged with shard headers
  client_events.jsonl(.gz)   merged client-side per-request event stream
                             (renamed from per_request.jsonl; plain when the
                             total is under PER_REQUEST_PLAIN_LIMIT_BYTES,
                             gzip at GZIP_COMPRESS_LEVEL otherwise)

Kept in place (skill / tooling contract):
  endpoints.json, flexlb_env.txt, client_env.json,
  mock_per_engine_timeseries.json.gz (A-split target),
  engine_events.jsonl (flipped to engine_events.jsonl.gz past the same
  size threshold as client_events.jsonl),
  load_client/server_latency.json (aggregate validity input; the skill's
  fetch_server_latency also reads it),
  flexlb_logs/pv.log (only produced with FLEXLB_PV_LOG=on).

Deleted after being merged: mock_engine.log, mock_engine_gc.log*,
master_info_before.json, master_info_after.json, master_prometheus_after.prom,
master_counters_timeseries.txt, mock_metrics_per_engine.prom,
master_prometheus_timeseries.prom, master_inflight_timeseries.jsonl,
process_usage_timeseries.txt, client_shard_*.stdout, client.stdout,
flexlb.log (run root), flexlb_logs/ (minus pv.log), load_client/shard_*/,
load_client/client_events.jsonl, slo_batch_analysis.stdout.
load_client/slo_batch_analysis.json is deleted only once its content is
embedded in client.json.

Every artifact is written to a ``.tmp`` sibling first and then os.replace()d
onto the final name, so an interrupted consolidation (kill, OOM, timeout)
never leaves a truncated file under a final name — the not-yet-deleted source
files stay authoritative and a re-run rebuilds everything.

The tool is idempotent and retro-runnable: re-running on an already
consolidated directory is a no-op (a regenerated slo_batch_analysis.json only
refreshes the slo_batch_summary keys, never blanks the merged one-shot
fields), and running it on a legacy (pre-consolidation) directory yields the
same layout. Missing inputs are skipped, never fatal, so the same code covers
partial runs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Match java_mock_stats key=value pairs (same shape as the MOCK_STAT_RE used by
# the analyzers, kept local so this tool has no cross-module import). Keys may
# carry digits (decode_exec_p50 / prefill_exec_p95 from mock-engine 4b14e05+);
# dropping them here used to zero out the exec-window percentiles in mock.json
# (aggregate reads mock.json once the legacy mock_engine.log is consolidated
# away, so the five-latency chart would silently degrade to all-zero exec
# series). The verbatim lines also stay available in mock.log.
STAT_KV_RE = re.compile(r"([a-z_][a-z_0-9]*)=(-?\d+(?:\.\d+)?)")
PROMETHEUS_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[-+\deE.]+)(\s+\d+)?\s*$"
)
# flexlb_env.txt lines look like:   'DOMAIN_ADDRESS:mock.prefill.hosts.address=host:1,host:2' \
ENV_FILE_LINE_RE = re.compile(r"^\s*'?([^=']+=[^']*)'?\s*\\?\s*$")
# Separator comment the per-second pollers prefix each sample with.
PROM_GROUP_TS_RE = re.compile(r"^#\s*ts=(\d+)\s*$")
# process_usage_timeseries.txt lines look like:
#   'ts_epoch_ms=1756... label=mock pid=123 cpu_pct=12.5 rss_kb=345600 etime=03:45'
PROCESS_USAGE_LINE_RE = re.compile(
    r"^ts_epoch_ms=(?P<ts>\d+)\s+label=(?P<label>\S+)\s+pid=(?P<pid>\d+)\s+"
    r"cpu_pct=(?P<cpu>[-+]?\d+(?:\.\d+)?)\s+rss_kb=(?P<rss>[-+]?\d+)\s+"
    r"etime=(?P<etime>\S+)\s*$"
)

# Below this total size the merged per-request stream stays plain text (the
# uniform-mode runs are small; gzip would only add an unpack step for readers).
PER_REQUEST_PLAIN_LIMIT_BYTES = 10 * 1024 * 1024
# Level 6 roughly halves gzip wall time vs the default 9 for <5% worse JSONL
# compression — consolidation must stay well inside the eval skill's
# DURATION+180s overall timeout window even on slow remote hosts.
GZIP_COMPRESS_LEVEL = 6

MOCK_STATS_LINE_PREFIX = "java_mock_stats "


def warn(message: str) -> None:
    print(f"[consolidate] WARNING: {message}", file=sys.stderr)


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write via a .tmp sibling + os.replace so readers never see a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def write_gzip_json_atomic(path: Path, payload: list) -> None:
    """A-split companion: stream the big per-engine timeline into a gzip JSON
    file (json.dump writes incrementally — no multi-hundred-MB string is ever
    materialized in memory)."""
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(
        tmp, "wt", encoding="utf-8", compresslevel=GZIP_COMPRESS_LEVEL
    ) as sink:
        json.dump(payload, sink, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def merge_log_atomic(target: Path, fill) -> None:
    """Fill a merged log into a .tmp sibling, then atomically take the final name."""
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as sink:
        fill(sink)
    os.replace(tmp, target)


def parse_mock_stats_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    snapshots: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if MOCK_STATS_LINE_PREFIX not in line:
                continue
            snapshot = {
                key: float(value) if "." in value else int(value)
                for key, value in STAT_KV_RE.findall(line)
            }
            if snapshot:
                snapshots.append(snapshot)
    return snapshots


def parse_counter_timeseries(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            row = {
                key: float(value) if "." in value else int(value)
                for key, value in STAT_KV_RE.findall(line)
            }
            if row:
                rows.append(row)
    return rows


def parse_prometheus_file(path: Path) -> dict[str, float]:
    """Prometheus text exposition format -> flat dict.

    Key is the raw ``name{labels}`` string (or bare ``name``), value the float
    sample; later lines win on key collision. HELP/TYPE comment lines are
    skipped. Keeping the labels in the key makes the dispatch_reason_total
    breakdown recoverable by the analyzers without a nested schema.
    """
    if not path.is_file():
        return {}
    metrics: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = PROMETHEUS_SAMPLE_RE.match(line)
            if not match:
                continue
            try:
                value = float(match.group("value"))
            except ValueError:
                continue
            key = match.group("name") + (match.group("labels") or "")
            metrics[key] = value
    return metrics


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse JavaMockEngineCluster's env file ('KEY=VALUE' \\ lines)."""
    if not path.is_file():
        return {}
    env: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = ENV_FILE_LINE_RE.match(line)
            if not match:
                continue
            pair = match.group(1)
            key, sep, value = pair.partition("=")
            if sep:
                env[key] = value
    return env


def parse_grouped_prometheus_timeseries(path: Path) -> list[dict]:
    """Grouped prom text (``# ts=<epoch_ms>`` separators) -> [{ts, metrics}].

    The per-second pollers (run_online_eval.sh) append each HTTP sample after
    a ``# ts=`` comment line. Samples inside a group are parsed with
    PROMETHEUS_SAMPLE_RE into a flat ``{name{labels}: value}`` dict (later
    lines win on key collision, same rule as parse_prometheus_file); HELP/
    TYPE comments and lines before the first ``# ts=`` marker are skipped
    (a torn trailing sample simply never lands in a group).
    """
    if not path.is_file():
        return []
    groups: list[dict] = []
    current: dict | None = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            match = PROM_GROUP_TS_RE.match(line)
            if match:
                current = {"ts": int(match.group(1)), "metrics": {}}
                groups.append(current)
                continue
            if current is None or not line or line.startswith("#"):
                continue
            sample = PROMETHEUS_SAMPLE_RE.match(line)
            if not sample:
                continue
            try:
                value = float(sample.group("value"))
            except ValueError:
                continue
            key = sample.group("name") + (sample.group("labels") or "")
            current["metrics"][key] = value
    return groups


def parse_jsonl_timeseries(path: Path) -> list[dict]:
    """JSONL file with one JSON object per line -> list of objects.

    Each line is json.loads'd independently: a torn trailing line (the
    poller was killed mid-write) is dropped instead of failing the merge.
    """
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def parse_process_usage_file(path: Path) -> list[dict]:
    """process_usage_timeseries.txt -> [{ts_epoch_ms, label, pid, ...}].

    One kv line per (sample, pid) as written by the process usage poller;
    unparseable lines (e.g. a partially written last line) are skipped.
    """
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = PROCESS_USAGE_LINE_RE.match(line.strip())
            if not match:
                continue
            rows.append(
                {
                    "ts_epoch_ms": int(match.group("ts")),
                    "label": match.group("label"),
                    "pid": int(match.group("pid")),
                    "cpu_pct": float(match.group("cpu")),
                    "rss_kb": int(match.group("rss")),
                    "etime": match.group("etime"),
                }
            )
    return rows


def load_param_json_file(path_str: str | None) -> dict | list | None:
    """Read a JSON config file referenced by a run_meta params path entry.

    Returns None when the path is empty, the file is missing, or the content
    is not valid JSON — run_meta.json then records null for that key.
    """
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, (dict, list)) else None


def fetch_final_snapshot(http_port: int | None) -> dict | None:
    """Best-effort /snapshot from the live mock control plane (base port - 1)."""
    if not http_port:
        return None
    url = f"http://127.0.0.1:{http_port}/snapshot"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def endpoints_summary(endpoints: dict) -> dict:
    if not endpoints:
        return {}
    engines = endpoints.get("engines", [])
    return {
        "prefill_domain": endpoints.get("prefill_domain"),
        "decode_domain": endpoints.get("decode_domain"),
        "env": endpoints.get("env", {}),
        "engine_count": len(engines) if isinstance(engines, list) else 0,
        "engines": (
            [
                {
                    key: engine.get(key)
                    for key in (
                        "name",
                        "role",
                        "grpc_port",
                        "http_port",
                        "grpc_addr",
                        "http_addr",
                    )
                }
                for engine in engines
                if isinstance(engine, dict)
            ]
            if isinstance(engines, list)
            else []
        ),
    }


def count_jsonl_rows(paths: list[Path]) -> int:
    """Non-blank line count across JSONL sources (per_request_source metadata).

    Phase B: replaces the streaming PerSecondAggregator — client.json's
    per_second timeline had no consumers left (aggregate_canvas_run.py
    recomputes per_second from the run-root client_events.jsonl itself, and
    the canvas report reads the aggregate), so
    only the cheap row-count metadata survives. Counts lines without
    json.loads: a single pass, no per-row parsing.
    """
    total = 0
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
            total += sum(1 for line in stream if line.strip())
    return total


def copy_stream(source: Path, sink) -> None:
    with source.open("rb") as stream:
        shutil.copyfileobj(stream, sink)


def append_section(sink, header: str, paths: list[Path]) -> list[Path]:
    """Append each existing path under a labelled separator header."""
    appended: list[Path] = []
    for path in paths:
        if path.is_file():
            sink.write(f"\n===== {header}: {path.name} =====\n".encode("utf-8"))
            copy_stream(path, sink)
            appended.append(path)
    return appended


def collect_per_request_sources(run_dir: Path, load_client: Path) -> list[Path]:
    """Mergeable per-request sources (shard dirs / single-worker file).

    The run-root merged files (client_events.jsonl / .gz — renamed from
    per_request.jsonl) are deliberately NOT listed: they are consolidation
    outputs, not inputs, so a re-run never re-merges (or deletes) its own
    output.
    """
    sources: list[Path] = []
    if load_client.is_dir():
        sources.extend(sorted(load_client.glob("shard_*/client_events.jsonl")))
        single = load_client / "client_events.jsonl"
        if single.is_file() and not sources:
            sources.append(single)
    return sources


def build_slo_summary(slo: dict) -> dict:
    """slo_batch_analysis.json -> the master.json slo_batch_summary fields."""
    slo_decisions = (
        slo.get("decisions", {}) if isinstance(slo.get("decisions"), dict) else {}
    )
    return {
        "decisions_count": slo_decisions.get("count"),
        "decision_reasons": slo_decisions.get("reasons", {}),
        "invariant_violation_count": slo_decisions.get("invariant_violation_count"),
        "completions_count": (
            slo.get("completions", {}).get("count")
            if isinstance(slo.get("completions"), dict)
            else None
        ),
        "mock_stats_samples": (
            slo.get("mock", {}).get("stats_samples")
            if isinstance(slo.get("mock"), dict)
            else None
        ),
        "test_valid": (
            slo.get("master", {}).get("test_valid")
            if isinstance(slo.get("master"), dict)
            else None
        ),
    }


def consolidate(
    run_dir: Path, params: dict[str, str], mock_http_port: int | None
) -> dict:
    report: dict[str, list[str]] = {"created": [], "deleted": [], "kept": []}
    load_client = run_dir / "load_client"
    flexlb_logs = run_dir / "flexlb_logs"
    deleted: list[Path] = []
    # m7: sweep stale *.tmp siblings left behind by an interrupted earlier
    # consolidation (atomic writes os.replace() the tmp away on success; only
    # a killed process leaves one behind). A stale tmp is never a valid input.
    for stale in sorted(run_dir.glob("*.tmp")):
        try:
            stale.unlink()
            report["deleted"].append(str(stale.relative_to(run_dir)))
        except OSError:
            pass

    # ---- run_meta.json -----------------------------------------------------
    endpoints = load_json(run_dir / "endpoints.json")
    existing_meta = load_json(run_dir / "run_meta.json")
    effective_params = params or existing_meta.get("params", {})
    # Config archive (G7): embed the full performance / process-config JSON
    # referenced by the params snapshot. Files live outside the run dir, so a
    # re-run re-reads them; when they have since disappeared the previously
    # embedded value is kept (same preservation rule as flexlb_env/params —
    # a re-run must never blank one-shot data).
    performance_json = load_param_json_file(effective_params.get("performance_file"))
    if performance_json is None:
        performance_json = existing_meta.get("performance_json")
    process_config_json = load_param_json_file(
        effective_params.get("process_config_file")
    )
    if process_config_json is None:
        process_config_json = existing_meta.get("process_config_json")
    # G5: per-second CPU/RSS timeline of the mock / master / client JVMs.
    # process_usage_timeseries.txt is a one-shot source (deleted below once
    # merged), so a re-run falls back to the previously embedded rows.
    process_usage = parse_process_usage_file(
        run_dir / "process_usage_timeseries.txt"
    ) or existing_meta.get("process_usage", [])
    run_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Preserve the previous snapshot when re-run without --param values
        # (otherwise the rewrite would blank the params the run started with).
        "params": effective_params,
        "flexlb_env": parse_env_file(run_dir / "flexlb_env.txt")
        or existing_meta.get("flexlb_env", {}),
        # M9: the JavaLoadClient env effective values (36 vars) snapshotted
        # at client launch; same status as flexlb_env — the source file stays
        # in place and the value is re-read on re-runs.
        "client_env": load_json(run_dir / "client_env.json")
        or existing_meta.get("client_env", {}),
        "endpoints": endpoints_summary(endpoints) or existing_meta.get("endpoints", {}),
        "performance_json": performance_json,
        "process_config_json": process_config_json,
        "process_usage": process_usage,
    }
    write_json_atomic(run_dir / "run_meta.json", run_meta)
    report["created"].append("run_meta.json")

    # ---- mock.json / mock.log ---------------------------------------------
    mock_log_path = run_dir / "mock_engine.log"
    mock_json_path = run_dir / "mock.json"
    mock_per_engine_path = run_dir / "mock_metrics_per_engine.prom"
    per_engine_gz_path = run_dir / "mock_per_engine_timeseries.json.gz"
    # Idempotency: when the source log is gone but mock.json already exists,
    # keep the existing file instead of overwriting it with an empty timeline.
    if (
        mock_log_path.is_file()
        or mock_per_engine_path.is_file()
        or not mock_json_path.is_file()
    ):
        stats = parse_mock_stats_file(mock_log_path)
        # Keep the previously captured snapshot when the control plane is
        # unreachable (re-run after the mock cluster exited): the old value is
        # the only record of the final cluster state. final_snapshot_source
        # records WHICH path produced the value so downstream reports can
        # flag utilization/engine-terminal data as stale (fallback) or absent
        # (missing) instead of silently trusting it.
        live_snapshot = fetch_final_snapshot(mock_http_port)
        final_snapshot = live_snapshot
        final_snapshot_source = "live"
        if final_snapshot is None:
            prior_snapshot = load_json(mock_json_path).get("final_snapshot")
            final_snapshot_source = "fallback" if prior_snapshot else "missing"
            final_snapshot = prior_snapshot
        # A-split: the per-engine timeline leaves mock.json and becomes its
        # own gzip file — at scale (1250 engines x 120s) the embedded
        # per_engine_timeseries key alone approaches 1GB of pretty-printed
        # JSON. mock.json keeps a pointer (per_engine_file) + sample count.
        existing_mock = load_json(mock_json_path)
        per_engine_ts = parse_grouped_prometheus_timeseries(mock_per_engine_path)
        if not per_engine_ts and not per_engine_gz_path.is_file():
            # One-shot .prom already merged away (re-run path), no split file
            # yet: migrate the legacy embedded key when present so even an
            # old fat mock.json gets slimmed on the next consolidation.
            legacy = existing_mock.get("per_engine_timeseries")
            if isinstance(legacy, list) and legacy:
                per_engine_ts = legacy
        per_engine_count = len(per_engine_ts) if per_engine_ts else None
        if per_engine_ts:
            write_gzip_json_atomic(per_engine_gz_path, per_engine_ts)
            report["created"].append(per_engine_gz_path.name)
        elif per_engine_gz_path.is_file() and per_engine_count is None:
            per_engine_count = existing_mock.get("per_engine_sample_count")
        mock_payload = {
            "stats_sample_count": len(stats),
            "stats": stats,
            "final_snapshot": final_snapshot,
            "final_snapshot_source": final_snapshot_source,
            "endpoints_summary": endpoints_summary(endpoints),
            # G1 per-second per-engine prometheus timeline, A-split into its
            # own gzip file ([{ts, metrics: {name{labels}: value}}] groups).
            "per_engine_file": (
                per_engine_gz_path.name if per_engine_gz_path.is_file() else None
            ),
            "per_engine_sample_count": per_engine_count,
        }
        write_json_atomic(mock_json_path, mock_payload)
        report["created"].append("mock.json")

    gc_logs = sorted(run_dir.glob("mock_engine_gc.log*"))
    if mock_log_path.is_file() or gc_logs:

        def fill_mock_log(sink) -> None:
            # mock_engine.log stays the verbatim prefix (tail-friendly).
            if mock_log_path.is_file():
                copy_stream(mock_log_path, sink)
                deleted.append(mock_log_path)
            for path in append_section(sink, "mock_engine_gc.log", gc_logs):
                deleted.append(path)

        merge_log_atomic(run_dir / "mock.log", fill_mock_log)
        for path in gc_logs:
            if path not in deleted:
                deleted.append(path)
        report["created"].append("mock.log")

    # ---- master.json / master.log ------------------------------------------
    slo_path = load_client / "slo_batch_analysis.json"
    slo = load_json(slo_path)
    # SLO freshness gate: slo_batch_analysis.json is regenerable at any time
    # by re-running analyze_slo_batch.py, so a stale leftover from an older
    # run would silently poison slo_batch_summary. The analysis is only
    # trusted when its mtime is >= the run's client_events.jsonl mtime (i.e.
    # it was produced from THIS run's data; the JSON key keeps the legacy
    # per_request_mtime name — canvas_report_gen.py renders it).
    per_request_path = load_client / "client_events.jsonl"
    slo_integrity = None
    if slo_path.is_file() and per_request_path.is_file():
        slo_mtime = slo_path.stat().st_mtime
        per_request_mtime = per_request_path.stat().st_mtime
        slo_integrity = {
            "slo_mtime": slo_mtime,
            "per_request_mtime": per_request_mtime,
            "fresh": slo_mtime >= per_request_mtime,
        }
    master_json_path = run_dir / "master.json"
    # Idempotency considers ONE-SHOT sources only. The slo file can be
    # regenerated at any time by re-running analyze_slo_batch.py after
    # consolidation; treating it as a rebuild trigger would blank
    # counters/prometheus/master_info (their sources are already merged away
    # and unrecoverable). A fresh slo file only refreshes slo_batch_summary.
    master_one_shot_sources = [
        run_dir / "master_counters_timeseries.txt",
        run_dir / "master_prometheus_after.prom",
        run_dir / "master_info_before.json",
        run_dir / "master_info_after.json",
        run_dir / "master_prometheus_timeseries.prom",
        run_dir / "master_inflight_timeseries.jsonl",
    ]
    if (
        any(path.is_file() for path in master_one_shot_sources)
        or not master_json_path.is_file()
    ):
        master_payload = {
            "counters_timeseries": parse_counter_timeseries(
                run_dir / "master_counters_timeseries.txt"
            ),
            # G3: per-second filtered flexlb_app_*/JVM prometheus timeline
            # (same grouped layout as mock per_engine_timeseries).
            "prometheus_timeseries": parse_grouped_prometheus_timeseries(
                run_dir / "master_prometheus_timeseries.prom"
            ),
            # G4: per-second inflight snapshots (one JSON object per line,
            # {"ts_epoch_ms": ..., "inflight": {...}}).
            "inflight_timeseries": parse_jsonl_timeseries(
                run_dir / "master_inflight_timeseries.jsonl"
            ),
            "prometheus_after": parse_prometheus_file(
                run_dir / "master_prometheus_after.prom"
            ),
            "master_info_before": load_json(run_dir / "master_info_before.json"),
            "master_info_after": load_json(run_dir / "master_info_after.json"),
            "slo_batch_summary": build_slo_summary(slo) if slo else {},
            "slo_integrity": slo_integrity,
        }
        write_json_atomic(master_json_path, master_payload)
        report["created"].append("master.json")
    elif slo:
        # Key-level incremental refresh: only slo_batch_summary tracks the
        # freshly regenerated slo file; the merged one-shot fields stay put.
        existing_master = load_json(master_json_path)
        if existing_master:
            existing_master["slo_batch_summary"] = build_slo_summary(slo)
            if slo_integrity is not None:
                existing_master["slo_integrity"] = slo_integrity
            write_json_atomic(master_json_path, existing_master)
            report["created"].append("master.json")

    master_log_sources: list[tuple[str, list[Path]]] = [
        (
            "application.log",
            (
                sorted(flexlb_logs.glob("application.log*"))
                if flexlb_logs.is_dir()
                else []
            ),
        ),
        (
            "flexlb.log",
            sorted(flexlb_logs.glob("flexlb.log*")) if flexlb_logs.is_dir() else [],
        ),
        (
            "sync.log",
            sorted(flexlb_logs.glob("sync.log*")) if flexlb_logs.is_dir() else [],
        ),
        (
            "sync_consistency.log",
            (
                sorted(flexlb_logs.glob("sync_consistency.log*"))
                if flexlb_logs.is_dir()
                else []
            ),
        ),
        ("master_stdout_flexlb.log", [run_dir / "flexlb.log"]),
    ]
    has_master_logs = any(
        path.is_file() for _, paths in master_log_sources for path in paths
    )
    if has_master_logs:

        def fill_master_log(sink) -> None:
            first = True
            for header, paths in master_log_sources:
                for path in paths:
                    if not path.is_file():
                        continue
                    if first:
                        # The application.log prefix stays verbatim (no header).
                        copy_stream(path, sink)
                        first = False
                    else:
                        sink.write(
                            f"\n===== {header}: {path.name} =====\n".encode("utf-8")
                        )
                        copy_stream(path, sink)
                    deleted.append(path)

        merge_log_atomic(run_dir / "master.log", fill_master_log)
        report["created"].append("master.log")
        # pv.log only exists with FLEXLB_PV_LOG=on; keep it in place.
        for kept in sorted(flexlb_logs.glob("*.log*")) if flexlb_logs.is_dir() else []:
            if kept not in deleted:
                report["kept"].append(str(kept.relative_to(run_dir)))

    # ---- client_events merge (before shard cleanup, feeds client.json) -----
    per_request_sources = collect_per_request_sources(run_dir, load_client)
    row_count = 0
    if per_request_sources:
        total_bytes = sum(path.stat().st_size for path in per_request_sources)
        plain = total_bytes < PER_REQUEST_PLAIN_LIMIT_BYTES
        target = run_dir / (
            "client_events.jsonl" if plain else "client_events.jsonl.gz"
        )
        tmp = target.with_name(target.name + ".tmp")

        def fill_per_request(sink) -> None:
            for path in per_request_sources:
                copy_stream(path, sink)

        if plain:
            with tmp.open("wb") as sink:
                fill_per_request(sink)
        else:
            with gzip.open(tmp, "wb", compresslevel=GZIP_COMPRESS_LEVEL) as sink:
                fill_per_request(sink)
        os.replace(tmp, target)
        # Never leave both compression variants behind (a legacy run root may
        # carry the opposite sibling from an earlier consolidation decision).
        alternate = run_dir / (
            "client_events.jsonl.gz" if plain else "client_events.jsonl"
        )
        if alternate.is_file():
            alternate.unlink()
        row_count = count_jsonl_rows([target])
        report["created"].append(target.name)
    else:
        # Re-run on a consolidated dir: the merged run-root file is the only
        # row source left (read-only) — recover the row count from it.
        for name in ("client_events.jsonl", "client_events.jsonl.gz"):
            path = run_dir / name
            if path.is_file():
                row_count = count_jsonl_rows([path])
                break

    # ---- engine_events.jsonl gzip flip (run root) ---------------------------
    # The mock engine writes its per-rid event rows straight to the run root
    # (engine-side half of the multi-component JSONL event streams). The same
    # size-based plain/gzip policy as client_events keeps large replay runs
    # from exploding the run dir. Same liveness semantics as the mock_engine.log
    # merge above: consolidation runs while the engine process is still up,
    # but the client has exited and drained, so no new event rows are expected
    # — the autoflushed prefix is the complete stream.
    for name in ("engine_events.jsonl", "engine_events.jsonl.gz"):
        engine_events = run_dir / name
        if not engine_events.is_file():
            continue
        if name.endswith(".gz") or (
            engine_events.stat().st_size < PER_REQUEST_PLAIN_LIMIT_BYTES
        ):
            report["kept"].append(name)
        else:
            gz_target = run_dir / "engine_events.jsonl.gz"
            gz_tmp = gz_target.with_name(gz_target.name + ".tmp")
            with engine_events.open("rb") as src, gzip.open(
                gz_tmp, "wb", compresslevel=GZIP_COMPRESS_LEVEL
            ) as sink:
                shutil.copyfileobj(src, sink)
            os.replace(gz_tmp, gz_target)
            engine_events.unlink()
            report["created"].append(gz_target.name)
            report["deleted"].append(engine_events.name)
        break

    # ---- client.json / client.log -------------------------------------------
    # Phase B: load_client/summary.json no longer exists (the Java client
    # records raw rows only; aggregate_canvas_run.py is the single derived-
    # statistics source). client.json is now a small embedding document —
    # server_latency + slo_batch_analysis + per_request_source — seeded from
    # the existing client.json (re-run case) so merged-away sources survive.
    client_payload = dict(load_json(run_dir / "client.json"))
    # server_latency.json is kept in place (aggregate validity input; the
    # skill's fetch_server_latency reads that exact path) but is still
    # embedded into client.json for single-file readers.
    server_latency = load_json(load_client / "server_latency.json")
    if server_latency:
        client_payload["server_latency"] = server_latency
    elif not isinstance(client_payload.get("server_latency"), dict):
        client_payload["server_latency"] = {}
    if slo:
        client_payload["slo_batch_analysis"] = slo
    if per_request_sources:
        client_payload["per_request_source"] = {
            "shard_count": len(per_request_sources),
            "row_count": row_count,
        }
    elif "per_request_source" not in client_payload:
        # Re-run recovery mode (no legacy sources left): the previous
        # shard_count is the only record of the original worker layout.
        client_payload["per_request_source"] = {
            "shard_count": 0,
            "row_count": row_count,
        }
    write_json_atomic(run_dir / "client.json", client_payload)
    report["created"].append("client.json")

    shard_stdouts = sorted(run_dir.glob("client_shard_*.stdout"))
    single_stdout = run_dir / "client.stdout"
    if shard_stdouts:

        def fill_client_log(sink) -> None:
            for path in shard_stdouts:
                sink.write(f"\n===== {path.stem} =====\n".encode("utf-8"))
                copy_stream(path, sink)
                deleted.append(path)

        merge_log_atomic(run_dir / "client.log", fill_client_log)
        report["created"].append("client.log")
    elif single_stdout.is_file():
        # Single-worker runs: the one stdout becomes client.log verbatim.
        client_log = run_dir / "client.log"
        if client_log.is_file():
            client_log.unlink()
        shutil.move(str(single_stdout), str(client_log))
        report["created"].append("client.log")

    # ---- cleanup of merged sources -----------------------------------------
    # Deletion is bound to successful merges: slo_batch_analysis.json is only
    # removed once its content is embedded in client.json. server_latency.json
    # stays in place entirely (aggregate validity input + skill
    # fetch_server_latency contract).
    for name in (
        "master_info_before.json",
        "master_info_after.json",
        "master_prometheus_after.prom",
        "master_counters_timeseries.txt",
        "master_prometheus_timeseries.prom",
        "master_inflight_timeseries.jsonl",
        "mock_metrics_per_engine.prom",
        "process_usage_timeseries.txt",
    ):
        path = run_dir / name
        if path.is_file():
            deleted.append(path)
    if slo:
        deleted.append(slo_path)
    slo_stdout = run_dir / "slo_batch_analysis.stdout"
    if slo_stdout.is_file():
        deleted.append(slo_stdout)
    for path in per_request_sources:
        if path.is_file():
            deleted.append(path)
    if load_client.is_dir():
        for shard_dir in sorted(load_client.glob("shard_*")):
            shutil.rmtree(shard_dir, ignore_errors=True)

    for path in deleted:
        try:
            if path.is_file():
                path.unlink()
                report["deleted"].append(str(path.relative_to(run_dir)))
        except OSError:
            pass

    report["kept"].extend(
        [
            "endpoints.json",
            "flexlb_env.txt",
            "client_env.json",
            "mock_per_engine_timeseries.json.gz",
            "load_client/server_latency.json",
            "flexlb_profile.jfr",
        ]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="startup parameter to snapshot into run_meta.json (repeatable)",
    )
    parser.add_argument(
        "--mock-http-port",
        type=int,
        default=None,
        help="live mock control-plane port (base grpc port - 1); when reachable "
        "the final cluster /snapshot is embedded into mock.json",
    )
    args = parser.parse_args()

    params: dict[str, str] = {}
    for item in args.param:
        key, sep, value = item.partition("=")
        if sep:
            params[key] = value
        else:
            warn(f"--param {item!r} has no '=' separator; ignored")

    report = consolidate(args.run_dir, params, args.mock_http_port)
    print(f"[consolidate] run_dir={args.run_dir}")
    for section in ("created", "deleted", "kept"):
        entries = report.get(section) or []
        print(f"[consolidate] {section}: {len(entries)}")
        for entry in entries:
            print(f"  - {entry}")


if __name__ == "__main__":
    main()
