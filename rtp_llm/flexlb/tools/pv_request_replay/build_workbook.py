#!/usr/bin/env python3
"""Build a request-level FlexLB PV / WorkerStatus investigation workbook.

Inputs may be raw ``pv.log`` snapshots or SLS CSV exports where the JSON PV
record lives in ``content``.  It joins the three PV event families by
``(FlexLB instance, request ID)``:

* FlexLB routing decision (route)
* KVCM predicted-vs-actual cache comparison (cache_hit_comparison)
* Prefill engine WorkerStatus telemetry (prefill_worker_status)

It deliberately does not invent Chat/Decode completion fields: that source is
not present in the supplied PV export.  Percentile bands are based on
``prefill_engine_ttft_ms`` (engine input enqueue -> first token).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import xlsxwriter


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
PV_MARKER = "pvLogger - "
LOG_TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
SOURCE_SUFFIXES = {".csv", ".log", ".txt", ".snapshot"}
# PDFUSION is the prefill-equivalent role in fused deployments.  It must be
# treated alongside PREFILL for route/cache/decision reconstruction, but DECODE
# remains deliberately excluded.
PREFILL_EQUIVALENT_ROLES = ("PREFILL", "PDFUSION")

PERCENTILE_COLORS = {
    "P0-P50": "C6EFCE",      # light green
    "P50-P90": "DDEBF7",     # light blue
    "P90-P95": "FFF2CC",     # light yellow
    "P95-P99": "FCE4D6",     # light orange
    "P99-P100": "E4DFEC",    # light purple
    "NO_TTFT": "FFC7CE",     # red / incomplete telemetry
}


@dataclass(frozen=True)
class PvSource:
    """One collected log snapshot and the FlexLB instance that produced it.

    Several snapshots can belong to the same instance.  Their events are
    intentionally de-duplicated by ``(instance, requestId)`` while identical
    request IDs from different instances remain independent.
    """

    path: Path
    instance: str


def _instance_from_path(path: Path, collection_root: Path | None = None) -> str:
    """Infer a stable instance label when no collector manifest is present."""

    name = path.name
    for suffix in (".snapshot", ".log", ".csv", ".txt"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    generic_names = {"pv", "pv.log", "snapshot", "content", "export", "pvlog", "raw"}
    if name.lower() not in generic_names:
        return name
    if collection_root is not None and path.parent != collection_root:
        return path.parent.name
    return path.parent.name or name


def _manifest_sources(manifest_path: Path) -> list[PvSource]:
    """Read the intentionally small manifest contract used by the collector.

    The reader accepts ``sources`` or ``snapshots`` lists so older collected
    bundles remain usable.  Every item must name a relative/absolute ``path``;
    ``instance`` is preferred, with ``pod`` and ``instance_id`` as aliases.
    """

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    items: Any = payload.get("sources") or payload.get("snapshots")
    if not isinstance(items, list):
        return []
    result: list[PvSource] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        instance = (item.get("instance") or item.get("pod")
                    or item.get("instance_id") or _instance_from_path(path, manifest_path.parent))
        result.append(PvSource(path.resolve(), str(instance)))
    return result


def discover_sources(input_path: str | Path) -> list[PvSource]:
    """Discover snapshots from a file or a collector output directory.

    A directory may contain ``collect_manifest.json`` (or legacy
    ``manifest.json``).  Without one, supported files
    are discovered recursively and the instance is inferred from file/parent
    names.  Results are deterministic and duplicate paths are removed.
    """

    root = Path(input_path).expanduser()
    if root.is_file():
        if root.name in {"collect_manifest.json", "manifest.json"}:
            sources = _manifest_sources(root)
            if not sources:
                raise ValueError(f"No usable snapshots in manifest: {root}")
            return sources
        return [PvSource(root.resolve(), _instance_from_path(root))]
    if not root.is_dir():
        raise FileNotFoundError(f"PV source does not exist: {root}")

    manifested = (_manifest_sources(root / "collect_manifest.json")
                  or _manifest_sources(root / "manifest.json"))
    discovered = manifested or [
        PvSource(path.resolve(), _instance_from_path(path, root))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    ]
    unique: dict[tuple[Path, str], PvSource] = {}
    for source in discovered:
        unique[(source.path, source.instance)] = source
    if not unique:
        raise FileNotFoundError(f"No pv.log/CSV snapshots found under: {root}")
    return list(unique.values())


def _normalize_boundary(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        try:
            value = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError(
                f"Invalid time {value!r}; use ISO-8601, for example "
                "2026-08-11T01:40:00+08:00"
            ) from error
    if value.tzinfo is None:
        value = value.replace(tzinfo=LOCAL_TZ)
    return value.astimezone(LOCAL_TZ)


def _boundary_ms(value: datetime | str | None) -> int | None:
    normalized = _normalize_boundary(value)
    return int(normalized.timestamp() * 1000) if normalized is not None else None


def _coerce_sources(sources: str | Path | PvSource | Sequence[str | Path | PvSource]) -> list[PvSource]:
    values: Sequence[str | Path | PvSource]
    if isinstance(sources, (str, Path, PvSource)):
        values = [sources]
    else:
        values = sources
    result: list[PvSource] = []
    for value in values:
        if isinstance(value, PvSource):
            result.append(PvSource(value.path.expanduser().resolve(), value.instance))
        else:
            result.extend(discover_sources(value))
    unique: dict[tuple[Path, str], PvSource] = {}
    for source in result:
        unique[(source.path, source.instance)] = source
    if not unique:
        raise ValueError("At least one PV source is required")
    return list(unique.values())


def as_number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else numeric


def non_negative(value: Any) -> int | float | None:
    numeric = as_number(value)
    return numeric if numeric is not None and numeric >= 0 else None


def epoch_ms_to_text(value: Any) -> str | None:
    numeric = as_number(value)
    if numeric is None or numeric <= 0:
        return None
    instant = datetime.fromtimestamp(numeric / 1000, tz=LOCAL_TZ)
    return instant.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(numeric) % 1000:03d}"


def log_time_from_content(content: str) -> str | None:
    match = LOG_TIME_RE.match(content)
    return match.group(1) if match else None


def parse_pv_record(content: str) -> tuple[str | None, dict[str, Any]] | None:
    marker_at = content.find(PV_MARKER)
    if marker_at < 0:
        return None
    raw_json = content[marker_at + len(PV_MARKER):].strip()
    try:
        return log_time_from_content(content), json.loads(raw_json)
    except json.JSONDecodeError:
        return None


def iter_pv_contents(source: PvSource):
    """Yield ``(instance, PV text)`` from an SLS CSV or raw pv.log.

    The collector keeps a byte-for-byte pv.log snapshot.  Supporting it here
    avoids a lossy intermediate export and retains the decision top-5 payload.
    An SLS export can contain several pods in one file; in that case its pod
    tag overrides the file-level instance hint so request IDs cannot cross-join
    between FlexLB instances.
    """

    with source.path.open(encoding="utf-8-sig", newline="") as source_file:
        first_line = source_file.readline()
        source_file.seek(0)
        try:
            header = next(csv.reader([first_line]))
        except csv.Error:
            header = []
        if "content" not in header:
            for content in source_file:
                yield source.instance, content
            return
        for raw in csv.DictReader(source_file):
            instance = (
                raw.get("__tag__:_pod_name_")
                or raw.get("__tag__:__hostname__")
                or source.instance
            )
            yield str(instance), raw.get("content", "")


def _role_items(route: dict[str, Any], field: str) -> list[dict[str, Any]]:
    if field == "server_status":
        response = route.get("response")
        values = response.get(field, []) if isinstance(response, dict) else []
    else:
        values = route.get(field, [])
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def get_prefill_equivalent_role(route: dict[str, Any] | None) -> str | None:
    """Resolve a supported route role, preserving PREFILL priority when both exist."""

    if not route:
        return None
    selection_reasons = route.get("selectionReasons", {})
    for role in PREFILL_EQUIVALENT_ROLES:
        if isinstance(selection_reasons, dict) and role in selection_reasons:
            return role
        for field in ("server_status", "cacheMatchSelections", "shortestTtftDecisions"):
            if any(item.get("role") == role for item in _role_items(route, field)):
                return role
    return None


def _prefill_equivalent_item(
    route: dict[str, Any] | None, field: str, role: str | None = None
) -> dict[str, Any]:
    if not route:
        return {}
    items = _role_items(route, field)
    for preferred_role in ((role,) if role else PREFILL_EQUIVALENT_ROLES):
        for item in items:
            if item.get("role") == preferred_role:
                return item
    if len(items) == 1 and not items[0].get("role"):
        return items[0]
    return {}


def get_prefill_server_status(route: dict[str, Any] | None) -> dict[str, Any]:
    return _prefill_equivalent_item(route, "server_status")


def get_route_cache_selection(route: dict[str, Any] | None) -> dict[str, Any]:
    return _prefill_equivalent_item(route, "cacheMatchSelections")


def get_prefill_decision(route: dict[str, Any] | None) -> dict[str, Any]:
    """Return the PREFILL/PDFUSION decision snapshot from a routing PV."""

    return _prefill_equivalent_item(route, "shortestTtftDecisions")


def yes_no_unknown(value: Any) -> str:
    """Render an optional boolean without collapsing missing into false."""

    if value is None:
        return ""
    return "YES" if bool(value) else "NO"


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percentile) - 1]


def ms_delta(later_ms: Any, earlier_ms: Any) -> int | None:
    later, earlier = non_negative(later_ms), non_negative(earlier_ms)
    if later is None or earlier is None or later <= 0 or earlier <= 0:
        return None
    delta = int(later - earlier)
    return delta if delta >= 0 else None


def pct(numerator: Any, denominator: Any) -> float | None:
    n, d = as_number(numerator), as_number(denominator)
    if n is None or d is None or d <= 0:
        return None
    return n / d * 100.0


def get_band(rank_pct: float) -> str:
    if rank_pct <= 50:
        return "P0-P50"
    if rank_pct <= 90:
        return "P50-P90"
    if rank_pct <= 95:
        return "P90-P95"
    if rank_pct <= 99:
        return "P95-P99"
    return "P99-P100"


def availability_text(route: bool, cache: bool, status: bool, has_ttft: bool) -> str:
    if not route:
        return "NO_ROUTE_PV"
    missing: list[str] = []
    if not cache:
        missing.append("NO_CACHE_COMPARISON")
    if not status:
        missing.append("NO_WORKER_STATUS")
    if status and not has_ttft:
        missing.append("NO_FIRST_TOKEN")
    return " | ".join(missing) if missing else "COMPLETE"


def route_success(route: dict[str, Any] | None) -> bool:
    if not route:
        return False
    response = route.get("response", {})
    return bool(route.get("success")) and response.get("code") == 200


def largest_phase(row: dict[str, Any]) -> tuple[str, float | None]:
    """Return a cause-neutral maximum among non-overlapping top-level phases.

    ``remote_kv_wait_ms`` is a component of ``scheduler_to_running_ms`` and is
    consequently not a peer in the top-level comparison.  It is still exposed
    in the workbook for scheduler sub-phase diagnosis.
    """

    candidates = {
        "INPUT_QUEUE": row.get("input_queue_wait_ms"),
        "SCHEDULER_TO_RUNNING": row.get("scheduler_to_running_ms"),
        "RUNNING_TO_FIRST_TOKEN": row.get("running_to_first_token_ms"),
    }
    valid = {name: float(value) for name, value in candidates.items()
             if isinstance(value, (int, float)) and value >= 0}
    if not valid:
        return "N/A", None
    name = max(valid, key=valid.get)
    engine_ttft = row.get("prefill_engine_ttft_ms")
    share = valid[name] / engine_ttft * 100 if engine_ttft and engine_ttft > 0 else None
    if name == "SCHEDULER_TO_RUNNING":
        remote = row.get("remote_kv_wait_ms")
        scheduler_total = row.get("scheduler_to_running_ms")
        if remote is not None and scheduler_total and remote / scheduler_total >= 0.5:
            name = "REMOTE_KV (within scheduler-to-running)"
        else:
            name = "SCHEDULER_WAIT (within scheduler-to-running)"
    return name, share


def observed_overlap(host_rows: list[dict[str, Any]]) -> None:
    """Annotate same-host predecessors that had not reached first token yet.

    This is descriptive overlap evidence only.  It does not prove that the
    predecessor was the reason a later request waited.
    """

    prior: list[dict[str, Any]] = []
    for index, row in enumerate(host_rows, start=1):
        decision_ms = row.get("request_time_ms")
        active = [candidate for candidate in prior
                  if decision_ms is not None
                  and candidate.get("first_token_time_ms") is not None
                  and candidate["first_token_time_ms"] > decision_ms]
        low_hit = [candidate for candidate in active
                   if candidate.get("actual_hit_rate_pct") is not None
                   and candidate["actual_hit_rate_pct"] < 90.0]
        row["host_sequence_no"] = index
        row["prior_before_first_token_count"] = len(active)
        row["prior_low_hit_before_first_token_count"] = len(low_hit)
        if active:
            latest = max(active, key=lambda candidate: candidate.get("request_time_ms") or -1)
            row["previous_request_id"] = latest.get("request_id")
            row["previous_actual_hit_rate_pct"] = latest.get("actual_hit_rate_pct")
            row["previous_uncache_tokens"] = latest.get("uncache_tokens")
            row["previous_prefill_engine_ttft_ms"] = latest.get("prefill_engine_ttft_ms")
            row["previous_first_token_time"] = latest.get("first_token_time")
        else:
            row["previous_request_id"] = None
            row["previous_actual_hit_rate_pct"] = None
            row["previous_uncache_tokens"] = None
            row["previous_prefill_engine_ttft_ms"] = None
            row["previous_first_token_time"] = None

        # The generic previous request can itself be a high-hit request.  Keep
        # the most recent *low-hit* overlapping predecessor separately so the
        # P99 investigation can navigate directly to the hypothesized source
        # of same-host interference instead of inferring it from a count.
        if low_hit:
            latest_low_hit = max(low_hit, key=lambda candidate: candidate.get("request_time_ms") or -1)
            row["low_hit_predecessor_request_id"] = latest_low_hit.get("request_id")
            row["low_hit_predecessor_actual_hit_rate_pct"] = latest_low_hit.get("actual_hit_rate_pct")
            row["low_hit_predecessor_uncache_tokens"] = latest_low_hit.get("uncache_tokens")
            row["low_hit_predecessor_prefill_engine_ttft_ms"] = latest_low_hit.get("prefill_engine_ttft_ms")
            row["low_hit_predecessor_first_token_time"] = latest_low_hit.get("first_token_time")
        else:
            row["low_hit_predecessor_request_id"] = None
            row["low_hit_predecessor_actual_hit_rate_pct"] = None
            row["low_hit_predecessor_uncache_tokens"] = None
            row["low_hit_predecessor_prefill_engine_ttft_ms"] = None
            row["low_hit_predecessor_first_token_time"] = None

        if low_hit:
            row["same_host_predecessor_evidence"] = "Earlier low-hit request had not reached first token"
        elif active:
            row["same_host_predecessor_evidence"] = "Earlier request had not reached first token"
        else:
            row["same_host_predecessor_evidence"] = "No earlier route record still before first token"
        prior.append(row)


def _store_latest(events: dict[tuple[str, str], tuple[str | None, dict[str, Any]]],
                  key: tuple[str, str], event_time: str | None,
                  record: dict[str, Any]) -> None:
    """Store a terminal/asynchronous event, de-duplicating overlapping snapshots."""

    previous = events.get(key)
    if previous is None or (event_time or "") >= (previous[0] or ""):
        events[key] = (event_time, record)


def _store_route(events: dict[tuple[str, str], tuple[str | None, dict[str, Any]]],
                 key: tuple[str, str], event_time: str | None,
                 record: dict[str, Any]) -> None:
    """Keep the latest route decision when overlapping snapshots repeat it."""

    previous = events.get(key)
    current_order = (as_number(record.get("requestTimeMs")) or -1, event_time or "")
    previous_order = ((as_number(previous[1].get("requestTimeMs")) or -1), previous[0] or "") if previous else None
    if previous_order is None or current_order >= previous_order:
        events[key] = (event_time, record)


def build_rows(sources: Sequence[PvSource], start: datetime | str | None = None,
               end: datetime | str | None = None
               ) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    """Correlate collected PV events and return request rows.

    Route inclusion uses ``requestTimeMs`` and the half-open interval
    ``[start, end)``.  Cache-comparison and WorkerStatus records are parsed
    without an event-time filter, allowing a collector's tail/grace window to
    complete requests routed just before ``end``.
    """

    start_ms = _boundary_ms(start)
    end_ms = _boundary_ms(end)
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError("start must be earlier than end")

    routes: dict[tuple[str, str], tuple[str | None, dict[str, Any]]] = {}
    cache_events: dict[tuple[str, str], tuple[str | None, dict[str, Any]]] = {}
    statuses: dict[tuple[str, str], tuple[str | None, dict[str, Any]]] = {}
    event_counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()

    for source in sources:
        event_counts["source_files"] += 1
        for event_instance, content in iter_pv_contents(source):
            parsed = parse_pv_record(content)
            if not parsed:
                continue
            event_time, record = parsed
            request_id = record.get("requestId")
            if not request_id:
                continue
            key = (event_instance, str(request_id))
            if "totalUs" in record:
                _store_route(routes, key, event_time, record)
                event_counts["route"] += 1
            elif record.get("event") == "cache_hit_comparison":
                _store_latest(cache_events, key, event_time, record)
                event_counts["cache_hit_comparison"] += 1
            elif record.get("event") == "prefill_worker_status":
                _store_latest(statuses, key, event_time, record)
                event_counts["prefill_worker_status"] += 1

    result: list[dict[str, Any]] = []
    # The main workbook is a FlexLB decision timeline.  A lone WorkerStatus
    # record without its matching routing PV has no decision time or selected
    # host evidence, so keep it out of the main request table rather than
    # presenting it as a normal request row.
    for flexlb_instance, request_id in sorted(routes):
        key = (flexlb_instance, request_id)
        route_event_time, route = routes.get(key, (None, None))
        cache_event_time, cache = cache_events.get(key, (None, None))
        status_event_time, status = statuses.get(key, (None, None))
        request_time_ms = non_negative(route.get("requestTimeMs") if route else None)
        if start_ms is not None and (request_time_ms is None or request_time_ms < start_ms):
            event_counts["route_outside_window"] += 1
            continue
        if end_ms is not None and (request_time_ms is None or request_time_ms >= end_ms):
            event_counts["route_outside_window"] += 1
            continue
        server_status = get_prefill_server_status(route)
        cache_selection = get_route_cache_selection(route)
        decision = get_prefill_decision(route)
        decision_workers = decision.get("workers", []) if isinstance(decision.get("workers", []), list) else []
        cache_affinity_decision = (decision.get("cacheAffinityDecision", {})
                                   if isinstance(decision.get("cacheAffinityDecision", {}), dict) else {})

        input_enqueue_ms = non_negative(status.get("inputQueueEnqueueTimeMs") if status else None)
        input_drain_ms = non_negative(status.get("inputQueueDrainTimeMs") if status else None)
        first_token_ms = non_negative(status.get("firstTokenTimeMs") if status else None)

        input_tokens = as_number(cache.get("inputTokens") if cache else None)
        if input_tokens is None:
            input_tokens = as_number(route.get("inputIdsCount") if route else None)
        if input_tokens is None:
            input_tokens = as_number(route.get("seqLen") if route else None)

        route_predicted_hit_tokens = as_number(cache_selection.get("hitCacheTokens"))
        predicted_hit_tokens = as_number(cache.get("kvcm", {}).get("hit") if cache else None)
        if predicted_hit_tokens is None:
            predicted_hit_tokens = route_predicted_hit_tokens
        actual_hit_tokens = as_number(cache.get("actual", {}).get("hit") if cache else None)
        hbm_local_match_tokens = non_negative(status.get("hbmLocalMatchTokens") if status else None)
        remote_kv_added_match_tokens = non_negative(status.get("remoteKvAddedMatchTokens") if status else None)

        actual_hit_rate = pct(actual_hit_tokens, input_tokens)
        predicted_hit_rate = pct(predicted_hit_tokens, input_tokens)
        uncache_tokens = int(input_tokens - actual_hit_tokens) if input_tokens is not None and actual_hit_tokens is not None else None
        actual_minus_predicted_tokens = (int(actual_hit_tokens - predicted_hit_tokens)
                                         if actual_hit_tokens is not None and predicted_hit_tokens is not None else None)
        actual_minus_predicted_pp = (actual_hit_rate - predicted_hit_rate
                                     if actual_hit_rate is not None and predicted_hit_rate is not None else None)
        hbm_plus_remote_tokens = (int(hbm_local_match_tokens + remote_kv_added_match_tokens)
                                  if hbm_local_match_tokens is not None and remote_kv_added_match_tokens is not None else None)
        actual_minus_hbm_remote_tokens = (int(actual_hit_tokens - hbm_plus_remote_tokens)
                                          if actual_hit_tokens is not None and hbm_plus_remote_tokens is not None else None)

        prefill_engine_ttft_ms = ms_delta(first_token_ms, input_enqueue_ms)
        route_to_first_token_ms = ms_delta(first_token_ms, request_time_ms)
        route_to_engine_enqueue_ms = ms_delta(input_enqueue_ms, request_time_ms)

        worker = ((status or {}).get("workerIp")
                  or server_status.get("server_ip")
                  or (cache or {}).get("worker")
                  or cache_selection.get("selectedIp"))
        selected_snapshot = next(
            (candidate for candidate in decision_workers
             if isinstance(candidate, dict) and candidate.get("selected")),
            None,
        )
        if selected_snapshot is None and worker:
            selected_snapshot = next(
                (candidate for candidate in decision_workers
                 if isinstance(candidate, dict) and candidate.get("ip") == worker),
                {},
            )
        selected_snapshot = selected_snapshot or {}
        selection_reasons = (route or {}).get("selectionReasons", {})
        selection_reason = next(
            (
                selection_reasons.get(role)
                for role in PREFILL_EQUIVALENT_ROLES
                if isinstance(selection_reasons, dict) and selection_reasons.get(role)
            ),
            "N/A",
        )
        selection_counts[selection_reason] += 1
        scheduler_wait = non_negative(status.get("schedulerWaitMs") if status else None)
        remote_wait = non_negative(status.get("remoteKvWaitMs") if status else None)
        scheduler_to_running = non_negative(status.get("schedulerToRunningMs") if status else None)
        scheduler_identity_delta = (scheduler_to_running - scheduler_wait - remote_wait
                                    if scheduler_to_running is not None and scheduler_wait is not None
                                    and remote_wait is not None else None)

        row: dict[str, Any] = {
            "request_id": request_id,
            "flexlb_instance": flexlb_instance,
            "prefill_host": worker or "UNKNOWN",
            # ``requestTimeMs`` is the decision-time ordering key.  The PV
            # logger timestamp can be a few milliseconds later/out of order
            # across logging threads, so do not present it as the timeline
            # field that users sort on.
            "route_log_time": epoch_ms_to_text(request_time_ms) or route_event_time,
            "route_pv_event_time": route_event_time,
            "request_time_ms": request_time_ms,
            "selection_reason": selection_reason,
            "decision_snapshot_status": ("TOP5" if decision_workers else "NOT_RECORDED"),
            "decision_time": epoch_ms_to_text(non_negative(decision.get("decisionTimeMs"))),
            "decision_routing_attempt": non_negative(decision.get("routingAttempt")),
            "decision_total_worker_count": non_negative(decision.get("totalWorkerCount")),
            "decision_candidate_worker_count": non_negative(decision.get("candidateWorkerCount")),
            "decision_snapshot_truncated": decision.get("snapshotTruncated"),
            "decision_cache_leader_ip_port": cache_affinity_decision.get("cacheLeaderIpPort"),
            "decision_shortest_ttft_ip_port": cache_affinity_decision.get("shortestTtftWorkerIpPort"),
            "decision_cache_lead_tokens": as_number(cache_affinity_decision.get("cacheLeadTokens")),
            "decision_extra_work_tokens": as_number(cache_affinity_decision.get("extraTtft")),
            "decision_tolerated_extra_work_tokens": as_number(cache_affinity_decision.get("toleratedExtraTtft")),
            "decision_outstanding_threshold_tokens": as_number(cache_affinity_decision.get("outstandingUncachedTokensThreshold")),
            "decision_cache_leader_outstanding_eligible": cache_affinity_decision.get("cacheLeaderOutstandingEligible"),
            "selected_snapshot_rank": non_negative(selected_snapshot.get("estimatedTtftRank")),
            "selected_snapshot_hit_rate_pct": as_number(selected_snapshot.get("requestHitRatePct")),
            "selected_snapshot_uncache_tokens": as_number(selected_snapshot.get("requestUncachedTokens")),
            "selected_snapshot_queue_time": as_number(selected_snapshot.get("queueTime")),
            "selected_snapshot_estimated_ttft": as_number(selected_snapshot.get("estimatedTtft")),
            "selected_snapshot_outstanding_uncached_tokens": as_number(selected_snapshot.get("outstandingUncachedTokens")),
            "selected_snapshot_outstanding_after_request_uncached_tokens": as_number(selected_snapshot.get("outstandingAfterRequestUncachedTokens")),
            "selected_snapshot_in_transit_waiting_uncached_tokens": as_number(selected_snapshot.get("inTransitAndWaitingUncachedTokens")),
            "selected_snapshot_tracked_running_remaining_prefill_tokens": as_number(selected_snapshot.get("trackedRunningRemainingPrefillTokens")),
            "selected_snapshot_engine_waiting_uncached_tokens": as_number(selected_snapshot.get("engineWaitingUncachedTokens")),
            "selected_snapshot_engine_running_remaining_prefill_tokens": as_number(selected_snapshot.get("engineRunningRemainingPrefillTokens")),
            "route_response_code": route.get("response", {}).get("code") if route else None,
            "prefill_route_code": server_status.get("code") if server_status else None,
            "route_success": route_success(route),
            "telemetry_status": availability_text(bool(route), bool(cache), bool(status), prefill_engine_ttft_ms is not None),
            "prefill_engine_ttft_ms": prefill_engine_ttft_ms,
            "route_to_first_token_ms": route_to_first_token_ms,
            "route_to_engine_enqueue_ms": route_to_engine_enqueue_ms,
            # Internal join/order keys used for same-host overlap evidence;
            # human-readable timestamps are emitted in the visible columns.
            "input_queue_enqueue_time_ms": input_enqueue_ms,
            "input_queue_drain_time_ms": input_drain_ms,
            "first_token_time_ms": first_token_ms,
            "input_queue_enqueue_time": epoch_ms_to_text(input_enqueue_ms),
            "input_queue_drain_time": epoch_ms_to_text(input_drain_ms),
            "first_token_time": epoch_ms_to_text(first_token_ms),
            "input_queue_wait_ms": non_negative(status.get("inputQueueWaitMs") if status else None),
            "scheduler_wait_ms": scheduler_wait,
            "remote_kv_wait_ms": remote_wait,
            "scheduler_to_running_ms": scheduler_to_running,
            "running_to_first_token_ms": non_negative(status.get("runningToFirstTokenMs") if status else None),
            "scheduler_identity_delta_ms": scheduler_identity_delta,
            "input_tokens": input_tokens,
            "uncache_tokens": uncache_tokens,
            "predicted_hit_tokens": predicted_hit_tokens,
            "route_predicted_hit_tokens": route_predicted_hit_tokens,
            "actual_hit_tokens": actual_hit_tokens,
            "actual_minus_predicted_tokens": actual_minus_predicted_tokens,
            "predicted_hit_rate_pct": predicted_hit_rate,
            "actual_hit_rate_pct": actual_hit_rate,
            "actual_minus_predicted_pp": actual_minus_predicted_pp,
            "hbm_local_match_tokens": hbm_local_match_tokens,
            "remote_kv_added_match_tokens": remote_kv_added_match_tokens,
            "hbm_plus_remote_tokens": hbm_plus_remote_tokens,
            "actual_minus_hbm_remote_tokens": actual_minus_hbm_remote_tokens,
            "first_prefill_step_id": non_negative(status.get("firstPrefillStepId") if status else None),
            "last_prefill_step_id": non_negative(status.get("lastPrefillStepId") if status else None),
            "prefill_step_count": non_negative(status.get("prefillStepCount") if status else None),
            "prefill_nonfinal_chunk_tokens_min": non_negative(status.get("prefillNonfinalChunkTokensMin") if status else None),
            "prefill_nonfinal_chunk_tokens_max": non_negative(status.get("prefillNonfinalChunkTokensMax") if status else None),
            "cache_event_time": cache_event_time,
            "cache_state": cache.get("state") if cache else None,
            "worker_status_event_time": status_event_time,
            "flexlb_route_total_us": non_negative(route.get("totalUs") if route else None),
            "flexlb_arrival_ms": non_negative(route.get("arrivalMs") if route else None),
            "flexlb_hash_wait_us": non_negative(route.get("hashWaitUs") if route else None),
            "flexlb_hash_us": non_negative(route.get("hashUs") if route else None),
            "flexlb_cache_match_us": non_negative(route.get("cacheMatchUs") if route else None),
            "flexlb_predicted_prefill_time": as_number(server_status.get("prefill_time") if server_status else None),
            "_decision_workers": decision_workers,
        }
        row["time_per_uncached_token_ms"] = (prefill_engine_ttft_ms / uncache_tokens
                                             if prefill_engine_ttft_ms is not None and uncache_tokens and uncache_tokens > 0
                                             else None)
        row["scheduler_remote_share_pct"] = (remote_wait / scheduler_to_running * 100.0
                                             if remote_wait is not None and scheduler_to_running and scheduler_to_running > 0
                                             else None)
        result.append(row)

    by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result:
        by_host[row["prefill_host"]].append(row)
    for host_rows in by_host.values():
        host_rows.sort(key=lambda row: (row.get("request_time_ms") is None, row.get("request_time_ms") or 0, row["request_id"]))
        observed_overlap(host_rows)

    # Global rank/band assignment is intentionally based on engine TTFT, while
    # the main sheet remains ordered as a per-host decision-time timeline.
    ranked = [row for row in result if row.get("prefill_engine_ttft_ms") is not None and row.get("route_success")]
    ranked.sort(key=lambda row: (row["prefill_engine_ttft_ms"], row["request_id"]))
    for rank, row in enumerate(ranked, start=1):
        percent = rank / len(ranked) * 100.0
        row["prefill_ttft_percentile"] = f"{get_band(percent)} ({percent:.2f}%)"
        row["prefill_ttft_band"] = get_band(percent)
    for row in result:
        if "prefill_ttft_band" not in row:
            row["prefill_ttft_percentile"] = "NO_TTFT / telemetry incomplete"
            row["prefill_ttft_band"] = "NO_TTFT"
        phase, share = largest_phase(row)
        row["dominant_observed_phase"] = phase
        row["dominant_phase_share_pct"] = share
        if row.get("actual_hit_rate_pct") is not None and row["actual_hit_rate_pct"] >= 95.0 and row["prefill_ttft_band"] in {"P95-P99", "P99-P100"}:
            row["high_hit_tail_flag"] = "High-hit tail: inspect phases / same-host predecessor evidence"
        else:
            row["high_hit_tail_flag"] = ""
    return result, event_counts, selection_counts


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    width: float
    kind: str = "text"  # text, integer, ms, us, pct, rate, bool


LEFT_COLUMNS = [
    Column("request_id", "request_id", 38),
    Column("flexlb_instance", "flexlb_instance", 30),
    Column("prefill_host", "prefill_host", 17),
    Column("host_sequence_no", "host_sequence_no", 12, "integer"),
    Column("route_log_time", "route_log_time (decision)", 24),
    Column("selection_reason", "selection_reason", 20),
    Column("decision_snapshot_status", "decision_snapshot_status", 18),
    Column("decision_routing_attempt", "decision_routing_attempt", 18, "integer"),
    Column("decision_cache_leader_ip_port", "decision cache leader", 23),
    Column("decision_shortest_ttft_ip_port", "decision shortest TTFT", 23),
    Column("decision_cache_lead_tokens", "decision cache lead tokens", 24, "integer"),
    Column("decision_extra_work_tokens", "decision extra work tokens", 24, "integer"),
    Column("decision_tolerated_extra_work_tokens", "decision max extra work tokens", 29, "integer"),
    Column("decision_outstanding_threshold_tokens", "decision outstanding threshold", 28, "integer"),
    Column("decision_cache_leader_outstanding_eligible", "cache leader outstanding eligible", 29),
    Column("selected_snapshot_rank", "selected snapshot TTFT rank", 26, "integer"),
    Column("selected_snapshot_hit_rate_pct", "selected snapshot hit rate", 25, "pct"),
    Column("selected_snapshot_uncache_tokens", "selected snapshot request uncache", 30, "integer"),
    Column("selected_snapshot_queue_time", "selected snapshot queue work", 28, "integer"),
    Column("selected_snapshot_estimated_ttft", "selected snapshot estimated TTFT", 31, "integer"),
    Column("selected_snapshot_outstanding_uncached_tokens", "selected outstanding uncache", 28, "integer"),
    Column("selected_snapshot_outstanding_after_request_uncached_tokens", "selected outstanding after request", 33, "integer"),
    Column("selected_snapshot_engine_waiting_uncached_tokens", "selected engine waiting uncache", 31, "integer"),
    Column("selected_snapshot_engine_running_remaining_prefill_tokens", "selected engine RUNNING remaining", 33, "integer"),
    Column("route_response_code", "route_response_code", 16, "integer"),
    Column("telemetry_status", "telemetry_status", 28),
    Column("prefill_engine_ttft_ms", "prefill_engine_ttft_ms", 23, "ms"),
    Column("prefill_ttft_percentile", "prefill_ttft_percentile", 22),
    Column("route_to_first_token_ms", "route_to_first_token_ms", 23, "ms"),
    Column("route_to_engine_enqueue_ms", "route_to_engine_enqueue_ms", 25, "ms"),
    Column("dominant_observed_phase", "dominant_observed_phase", 35),
    Column("dominant_phase_share_pct", "dominant_phase_share_pct", 22, "rate"),
    Column("input_queue_wait_ms", "input_queue_wait_ms", 21, "ms"),
    Column("scheduler_wait_ms", "scheduler_wait_ms", 20, "ms"),
    Column("remote_kv_wait_ms", "remote_kv_wait_ms (scheduler subset)", 30, "ms"),
    Column("scheduler_to_running_ms", "scheduler_to_running_ms", 25, "ms"),
    Column("running_to_first_token_ms", "running_to_first_token_ms", 27, "ms"),
    Column("hbm_local_match_tokens", "hbm_local_match_tokens", 24, "integer"),
    Column("remote_kv_added_match_tokens", "remote_kv_added_match_tokens", 28, "integer"),
    Column("prefill_step_count", "prefill_step_count", 18, "integer"),
    Column("first_prefill_step_id", "first_prefill_step_id", 21, "integer"),
    Column("last_prefill_step_id", "last_prefill_step_id", 20, "integer"),
    Column("prefill_nonfinal_chunk_tokens_min", "prefill_nonfinal_chunk_tokens_min", 29, "integer"),
    Column("prefill_nonfinal_chunk_tokens_max", "prefill_nonfinal_chunk_tokens_max", 29, "integer"),
]

REFERENCE_AND_EVIDENCE_COLUMNS = [
    Column("input_tokens", "input_tokens", 16, "integer"),
    Column("uncache_tokens", "uncache_tokens", 18, "integer"),
    Column("time_per_uncached_token_ms", "time per uncached token (TTFT ms)", 31, "ms"),
    Column("actual_hit_rate_pct", "actual_hit_rate_pct", 20, "pct"),
    Column("predicted_hit_rate_pct", "predicted_hit_rate_pct", 22, "pct"),
    Column("actual_minus_predicted_pp", "actual_minus_predicted_pp", 25, "pct"),
    Column("predicted_hit_tokens", "predicted_hit_tokens", 21, "integer"),
    Column("route_predicted_hit_tokens", "route_predicted_hit_tokens", 25, "integer"),
    Column("actual_hit_tokens", "actual_hit_tokens", 18, "integer"),
    Column("actual_minus_predicted_tokens", "actual_minus_predicted_tokens", 27, "integer"),
    Column("hbm_plus_remote_tokens", "hbm + remote added tokens", 25, "integer"),
    Column("actual_minus_hbm_remote_tokens", "actual - (hbm + remote)", 26, "integer"),
    Column("scheduler_remote_share_pct", "remote KV share of scheduler→running", 32, "rate"),
    Column("input_queue_enqueue_time", "input_queue_enqueue_time", 25),
    Column("input_queue_drain_time", "input_queue_drain_time", 25),
    Column("first_token_time", "first_token_time", 25),
    Column("scheduler_identity_delta_ms", "scheduler_to_running - wait - remote", 32, "ms"),
    Column("previous_request_id", "previous_request_id (same host)", 39),
    Column("previous_actual_hit_rate_pct", "previous_actual_hit_rate_pct", 27, "pct"),
    Column("previous_uncache_tokens", "previous_uncache_tokens", 24, "integer"),
    Column("previous_prefill_engine_ttft_ms", "previous_prefill_engine_ttft_ms", 30, "ms"),
    Column("previous_first_token_time", "previous_first_token_time", 26),
    Column("prior_before_first_token_count", "prior before-first-token count", 27, "integer"),
    Column("prior_low_hit_before_first_token_count", "prior low-hit before-first-token count", 34, "integer"),
    Column("low_hit_predecessor_request_id", "low-hit predecessor request_id", 39),
    Column("low_hit_predecessor_actual_hit_rate_pct", "low-hit predecessor actual_hit_rate_pct", 34, "pct"),
    Column("low_hit_predecessor_uncache_tokens", "low-hit predecessor uncache_tokens", 31, "integer"),
    Column("low_hit_predecessor_prefill_engine_ttft_ms", "low-hit predecessor engine_ttft_ms", 34, "ms"),
    Column("low_hit_predecessor_first_token_time", "low-hit predecessor first_token_time", 31),
    Column("same_host_predecessor_evidence", "same_host_predecessor_evidence", 44),
    Column("high_hit_tail_flag", "high_hit_tail_flag", 48),
    Column("cache_event_time", "cache_event_time", 24),
    Column("cache_state", "cache_state", 14),
    Column("route_pv_event_time", "route_pv_event_time", 24),
    Column("worker_status_event_time", "worker_status_event_time", 27),
    Column("flexlb_route_total_us", "flexlb_route_total_us (not E2E)", 29, "us"),
    Column("flexlb_arrival_ms", "flexlb_arrival_ms", 18, "ms"),
    Column("flexlb_hash_wait_us", "flexlb_hash_wait_us", 21, "us"),
    Column("flexlb_hash_us", "flexlb_hash_us", 16, "us"),
    Column("flexlb_cache_match_us", "flexlb_cache_match_us", 23, "us"),
    Column("flexlb_predicted_prefill_time", "flexlb_predicted_prefill_time (routing prediction)", 39, "ms"),
]

ALL_COLUMNS = LEFT_COLUMNS + REFERENCE_AND_EVIDENCE_COLUMNS


def write_cell(worksheet: xlsxwriter.worksheet.Worksheet, row: int, col: int, value: Any,
               cell_format: xlsxwriter.format.Format) -> None:
    if value is None:
        worksheet.write_blank(row, col, None, cell_format)
    else:
        worksheet.write(row, col, value, cell_format)


def fill_row(worksheet: xlsxwriter.worksheet.Worksheet, row: int, last_col: int,
             cell_format: xlsxwriter.format.Format) -> None:
    for col in range(last_col + 1):
        worksheet.write_blank(row, col, None, cell_format)


def formats(workbook: xlsxwriter.Workbook) -> dict[str, xlsxwriter.format.Format]:
    return {
        "title": workbook.add_format({"bold": True, "font_size": 14, "font_color": "1F1F1F", "bg_color": "EAF2F8", "align": "left", "valign": "vcenter"}),
        "note": workbook.add_format({"font_color": "1F1F1F", "bg_color": "F5F9FC", "text_wrap": True, "valign": "top"}),
        "legend_label": workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1}),
        "header": workbook.add_format({"bold": True, "font_color": "FFFFFF", "bg_color": "1F4E78", "align": "center", "valign": "vcenter", "text_wrap": True, "border": 1, "border_color": "D9E2F3"}),
        "host": workbook.add_format({"bold": True, "font_color": "FFFFFF", "bg_color": "305496", "align": "left", "valign": "vcenter"}),
        "text": workbook.add_format({"valign": "top"}),
        "integer": workbook.add_format({"num_format": "#,##0", "valign": "top"}),
        "ms": workbook.add_format({"num_format": "#,##0", "valign": "top"}),
        "us": workbook.add_format({"num_format": "#,##0", "valign": "top"}),
        "pct": workbook.add_format({"num_format": '0.000"%"', "valign": "top"}),
        "rate": workbook.add_format({"num_format": '0.0"%"', "valign": "top"}),
        "summary_header": workbook.add_format({"bold": True, "font_color": "FFFFFF", "bg_color": "1F4E78", "align": "center", "valign": "vcenter", "text_wrap": True}),
        "summary_text": workbook.add_format({"valign": "top"}),
        "summary_int": workbook.add_format({"num_format": "#,##0", "valign": "top"}),
        "summary_ms": workbook.add_format({"num_format": "#,##0", "valign": "top"}),
        "summary_pct": workbook.add_format({"num_format": '0.0"%"', "valign": "top"}),
        "warning": workbook.add_format({"bold": True, "font_color": "9C0006", "bg_color": "FFC7CE", "text_wrap": True}),
    }


def write_requests_sheet(workbook: xlsxwriter.Workbook, rows: list[dict[str, Any]], threshold: dict[str, float | None]) -> None:
    worksheet = workbook.add_worksheet("Requests")
    fmt = formats(workbook)
    last_col = len(ALL_COLUMNS) - 1
    worksheet.hide_gridlines(2)
    worksheet.set_zoom(78)
    fill_row(worksheet, 0, last_col, fmt["title"])
    worksheet.write(0, 0, "FlexLB PV × cache comparison × Prefill WorkerStatus — host timeline (decision-time ascending)", fmt["title"])
    worksheet.set_row(0, 26)
    fill_row(worksheet, 1, last_col, fmt["note"])
    worksheet.write(1, 0, "Scope", fmt["note"])
    worksheet.write(1, 1, "This CSV has FlexLB route/cache and Prefill WorkerStatus only. It does NOT contain Chat/Decode completion or full Prefill finish time.", fmt["note"])
    worksheet.write(1, 2, "Percentile basis", fmt["note"])
    worksheet.write(1, 3, "P90/P95/P99 colors use prefill_engine_ttft_ms = firstTokenTimeMs - inputQueueEnqueueTimeMs; this is Prefill engine TTFT, not Chat E2E.", fmt["note"])
    worksheet.set_row(1, 38)

    legends = [
        ("P0-P50", f"P0–P50 ≤ {threshold['P50']:,} ms" if threshold["P50"] is not None else "P0–P50"),
        ("P50-P90", f"P50–P90 ≤ {threshold['P90']:,} ms" if threshold["P90"] is not None else "P50–P90"),
        ("P90-P95", f"P90–P95 ≤ {threshold['P95']:,} ms" if threshold["P95"] is not None else "P90–P95"),
        ("P95-P99", f"P95–P99 ≤ {threshold['P99']:,} ms" if threshold["P99"] is not None else "P95–P99"),
        ("P99-P100", "P99–P100 tail"),
    ]
    for col, (band, text) in enumerate(legends):
        legend_fmt = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "bg_color": PERCENTILE_COLORS[band], "border": 1})
        worksheet.write(2, col, text, legend_fmt)
    for col in range(len(legends), last_col + 1):
        worksheet.write_blank(2, col, None, fmt["legend_label"])
    worksheet.set_row(2, 22)

    for col, column in enumerate(ALL_COLUMNS):
        worksheet.write(3, col, column.label, fmt["header"])
        worksheet.set_column(col, col, column.width)
    worksheet.set_row(3, 38)
    worksheet.freeze_panes(4, 1)
    worksheet.autofilter(3, 0, 3 + len(rows) + len({row['prefill_host'] for row in rows}), last_col)

    formats_by_band = {
        band: {kind: workbook.add_format({
            "bg_color": color,
            "valign": "top",
            "num_format": {"integer": "#,##0", "ms": "#,##0", "us": "#,##0", "pct": '0.000"%"', "rate": '0.0"%"'}.get(kind, "General"),
        }) for kind in ("text", "integer", "ms", "us", "pct", "rate")}
        for band, color in PERCENTILE_COLORS.items()
    }
    error_formats = {
        kind: workbook.add_format({
            "bg_color": "FFC7CE", "font_color": "9C0006", "valign": "top",
            "num_format": {"integer": "#,##0", "ms": "#,##0", "us": "#,##0", "pct": '0.000"%"', "rate": '0.0"%"'}.get(kind, "General"),
        }) for kind in ("text", "integer", "ms", "us", "pct", "rate")
    }

    current_row = 4
    for host in sorted({row["prefill_host"] for row in rows}):
        # The physical sheet order is intentional: each visible host section
        # is a decision-time timeline, independent of the request-id ordering
        # used while parsing the source CSV.
        host_rows = sorted(
            (row for row in rows if row["prefill_host"] == host),
            key=lambda row: (row.get("request_time_ms") is None,
                             row.get("request_time_ms") or 0,
                             row["request_id"]),
        )
        tail_count = sum(row["prefill_ttft_band"] == "P99-P100" for row in host_rows)
        high_hit_tail_count = sum(bool(row["high_hit_tail_flag"]) for row in host_rows)
        fill_row(worksheet, current_row, last_col, fmt["host"])
        host_summary = [
            ("section", "Prefill host"),
            ("prefill_host", host),
            ("requests", len(host_rows)),
            ("p99_tail", tail_count),
            ("high_hit_tail", high_hit_tail_count),
        ]
        for offset, (key, value) in enumerate(host_summary):
            worksheet.write(current_row, offset * 2, key, fmt["host"])
            worksheet.write(current_row, offset * 2 + 1, value, fmt["host"])
        worksheet.set_row(current_row, 20)
        current_row += 1
        group_start = current_row
        for row in host_rows:
            band = row["prefill_ttft_band"]
            is_incomplete_or_error = (band == "NO_TTFT" or not row.get("route_success"))
            row_formats = error_formats if is_incomplete_or_error else formats_by_band[band]
            for col, column in enumerate(ALL_COLUMNS):
                kind = column.kind if column.kind in row_formats else "text"
                write_cell(worksheet, current_row, col, row.get(column.key), row_formats[kind])
            worksheet.set_row(current_row, 30)
            current_row += 1
        # Excel preserves the group outline; the visible host header remains
        # useful after import in Numbers even if it drops outline controls.
        if current_row > group_start:
            worksheet.set_row(group_start, None, None, {"level": 1})
            for outlined_row in range(group_start + 1, current_row):
                worksheet.set_row(outlined_row, None, None, {"level": 1})

    # Put the essential definitions next to the main data, not only in a
    # separate README, so a copied sheet does not lose interpretation.
    worksheet.set_comments_author("Codex")
    column_index = {column.key: index for index, column in enumerate(ALL_COLUMNS)}
    worksheet.write_comment(3, column_index["prefill_engine_ttft_ms"], "P bands use firstTokenTimeMs - inputQueueEnqueueTimeMs. This source has no Chat/Decode E2E completion.")
    worksheet.write_comment(3, column_index["hbm_local_match_tokens"], "HBM-local cache hits. Together with remote_kv_added_match_tokens it equals engine actual cache-hit tokens for joined rows.")
    worksheet.write_comment(3, column_index["remote_kv_wait_ms"], "A component of scheduler_to_running_ms; do not add remote_kv_wait_ms to scheduler_to_running_ms again.")
    worksheet.write_comment(3, column_index["running_to_first_token_ms"], "Measured from RUNNING to the first generated token. It can be high under chunked prefill / batch compute contention even on a high cache hit.")


def write_p99_focus_sheet(workbook: xlsxwriter.Workbook, rows: list[dict[str, Any]], threshold: dict[str, float | None]) -> None:
    worksheet = workbook.add_worksheet("P99 Focus")
    fmt = formats(workbook)
    visible_columns = [
        "request_id", "flexlb_instance", "prefill_host", "route_log_time", "selection_reason", "prefill_engine_ttft_ms",
        "prefill_ttft_percentile", "route_to_first_token_ms", "actual_hit_rate_pct", "uncache_tokens",
        "hbm_local_match_tokens", "remote_kv_added_match_tokens", "input_queue_wait_ms", "scheduler_wait_ms",
        "remote_kv_wait_ms", "scheduler_to_running_ms", "running_to_first_token_ms", "dominant_observed_phase",
        "dominant_phase_share_pct", "prefill_step_count", "prefill_nonfinal_chunk_tokens_max",
        "prior_before_first_token_count", "prior_low_hit_before_first_token_count", "previous_request_id",
        "previous_actual_hit_rate_pct", "previous_uncache_tokens", "low_hit_predecessor_request_id",
        "low_hit_predecessor_actual_hit_rate_pct", "low_hit_predecessor_uncache_tokens",
        "low_hit_predecessor_prefill_engine_ttft_ms", "same_host_predecessor_evidence", "high_hit_tail_flag",
        "telemetry_status",
    ]
    columns = [next(column for column in ALL_COLUMNS if column.key == key) for key in visible_columns]
    tail_rows = [row for row in rows if row["prefill_ttft_band"] in {"P95-P99", "P99-P100", "NO_TTFT"}]
    tail_rows.sort(key=lambda row: (row.get("prefill_engine_ttft_ms") is None, -(row.get("prefill_engine_ttft_ms") or -1), row["request_id"]))
    last_col = len(columns) - 1
    worksheet.hide_gridlines(2)
    worksheet.set_zoom(85)
    fill_row(worksheet, 0, last_col, fmt["title"])
    worksheet.write(0, 0, "P95/P99 Tail Focus - sorted by Prefill engine TTFT descending", fmt["title"])
    fill_row(worksheet, 1, last_col, fmt["note"])
    worksheet.write(1, 0, "Interpretation", fmt["note"])
    worksheet.write(1, 1, "Use this sheet to distinguish high cache hit from low wait: read input_queue_wait_ms, scheduler_to_running_ms, remote_kv_wait_ms, and running_to_first_token_ms.", fmt["note"])
    worksheet.write(1, 2, "Same-host predecessor", fmt["note"])
    worksheet.write(1, 3, "Correlation evidence, not causal proof.", fmt["note"])
    worksheet.set_row(1, 34)
    for col, column in enumerate(columns):
        worksheet.write(3, col, column.label, fmt["header"])
        worksheet.set_column(col, col, column.width)
    worksheet.set_row(3, 38)
    worksheet.freeze_panes(4, 1)
    worksheet.autofilter(3, 0, 3 + len(tail_rows), last_col)

    band_formats = {
        band: {kind: workbook.add_format({
            "bg_color": color, "valign": "top",
            "num_format": {"integer": "#,##0", "ms": "#,##0", "us": "#,##0", "pct": '0.000"%"', "rate": '0.0"%"'}.get(kind, "General"),
        }) for kind in ("text", "integer", "ms", "us", "pct", "rate")}
        for band, color in PERCENTILE_COLORS.items()
    }
    for out_row, row in enumerate(tail_rows, start=4):
        row_formats = band_formats[row["prefill_ttft_band"]]
        for col, column in enumerate(columns):
            write_cell(worksheet, out_row, col, row.get(column.key), row_formats[column.kind])
        worksheet.set_row(out_row, 30)


def write_decision_snapshot_sheet(workbook: xlsxwriter.Workbook, rows: list[dict[str, Any]]) -> None:
    """Write the recorded top-five candidate cut for each route decision.

    These fields are sampled by FlexLB at routing time.  They are intentionally
    separate from terminal WorkerStatus fields so the sheet cannot imply that a
    candidate's later completion state was known while making the decision.
    """

    worksheet = workbook.add_worksheet("Decision Snapshot Top5")
    fmt = formats(workbook)
    headers = [
        ("route_log_time (decision)", 24, "text", lambda row, candidate: row.get("route_log_time")),
        ("request_id", 39, "text", lambda row, candidate: row.get("request_id")),
        ("flexlb_instance", 30, "text", lambda row, candidate: row.get("flexlb_instance")),
        ("selected_prefill_host", 19, "text", lambda row, candidate: row.get("prefill_host")),
        ("prefill_engine_ttft_ms (terminal)", 30, "ms", lambda row, candidate: row.get("prefill_engine_ttft_ms")),
        ("TTFT percentile (terminal)", 24, "text", lambda row, candidate: row.get("prefill_ttft_percentile")),
        ("selection_reason", 27, "text", lambda row, candidate: row.get("selection_reason")),
        ("candidate_rank_by_estimated_TTFT", 30, "integer", lambda row, candidate: candidate.get("estimatedTtftRank")),
        ("candidate_ip", 18, "text", lambda row, candidate: candidate.get("ip")),
        ("candidate_port", 15, "integer", lambda row, candidate: candidate.get("port")),
        ("selected", 12, "text", lambda row, candidate: "YES" if candidate.get("selected") else ""),
        ("cache_leader", 15, "text", lambda row, candidate: "YES" if candidate.get("cacheLeader") else ""),
        ("shortest_TTFT", 15, "text", lambda row, candidate: "YES" if candidate.get("shortestTtftWorker") else ""),
        ("outstanding_guard_eligible", 27, "text", lambda row, candidate: yes_no_unknown(candidate.get("outstandingGuardEligible"))),
        ("request_hit_rate_pct", 22, "pct", lambda row, candidate: candidate.get("requestHitRatePct")),
        ("request_hit_cache_tokens", 24, "integer", lambda row, candidate: candidate.get("requestHitCacheTokens")),
        ("request_uncache_tokens", 23, "integer", lambda row, candidate: candidate.get("requestUncachedTokens")),
        ("request_prefill_work", 21, "integer", lambda row, candidate: candidate.get("requestPrefillTime")),
        ("queue_work_before_route", 23, "integer", lambda row, candidate: candidate.get("queueTime")),
        ("estimated_TTFT_work", 22, "integer", lambda row, candidate: candidate.get("estimatedTtft")),
        ("outstanding_uncache_before", 26, "integer", lambda row, candidate: candidate.get("outstandingUncachedTokens")),
        ("outstanding_uncache_after", 25, "integer", lambda row, candidate: candidate.get("outstandingAfterRequestUncachedTokens")),
        ("in_transit_waiting_task_count", 28, "integer", lambda row, candidate: candidate.get("inTransitAndWaitingTaskCount")),
        ("in_transit_waiting_uncache", 29, "integer", lambda row, candidate: candidate.get("inTransitAndWaitingUncachedTokens")),
        ("tracked_RUNNING_task_count", 28, "integer", lambda row, candidate: candidate.get("trackedRunningTaskCount")),
        ("tracked_RUNNING_remaining", 29, "integer", lambda row, candidate: candidate.get("trackedRunningRemainingPrefillTokens")),
        ("engine_WAITING_task_count", 27, "integer", lambda row, candidate: candidate.get("engineWaitingTaskCount")),
        ("engine_WAITING_uncache", 25, "integer", lambda row, candidate: candidate.get("engineWaitingUncachedTokens")),
        ("engine_RUNNING_task_count", 28, "integer", lambda row, candidate: candidate.get("engineRunningTaskCount")),
        ("engine_RUNNING_remaining", 29, "integer", lambda row, candidate: candidate.get("engineRunningRemainingPrefillTokens")),
        ("available_KV_cache_tokens", 27, "integer", lambda row, candidate: candidate.get("availableKvCacheTokens")),
        ("used_KV_cache_tokens", 23, "integer", lambda row, candidate: candidate.get("usedKvCacheTokens")),
        ("status_age_us", 18, "us", lambda row, candidate: candidate.get("statusAgeUs")),
        ("status_update_interval_us", 27, "us", lambda row, candidate: candidate.get("statusUpdateIntervalUs")),
        ("cache_age_us", 18, "us", lambda row, candidate: candidate.get("cacheAgeUs")),
    ]
    last_col = len(headers) - 1
    worksheet.hide_gridlines(2)
    worksheet.set_zoom(78)
    fill_row(worksheet, 0, last_col, fmt["title"])
    worksheet.write(0, 0, "FlexLB decision-time top-5 candidate snapshots", fmt["title"])
    fill_row(worksheet, 1, last_col, fmt["note"])
    worksheet.write(1, 0, "Scope", fmt["note"])
    worksheet.write(1, 1, "Every row is one candidate from the top-5 snapshot recorded at the route decision. queue/TTFT values are FlexLB token-work estimates, not observed wall-clock milliseconds.", fmt["note"])
    worksheet.write(1, 2, "Use", fmt["note"])
    worksheet.write(1, 3, "Compare the selected row with cache_leader/shortest_TTFT rows to inspect the candidate set visible at that exact decision instant.", fmt["note"])
    worksheet.set_row(1, 38)
    for col, (label, width, _, _) in enumerate(headers):
        worksheet.write(3, col, label, fmt["header"])
        worksheet.set_column(col, col, width)
    worksheet.set_row(3, 38)
    worksheet.freeze_panes(4, 2)

    number_formats = {"integer": "#,##0", "ms": "#,##0", "us": "#,##0", "pct": '0.000"%"'}
    band_formats = {
        band: {
            kind: workbook.add_format({"bg_color": color, "valign": "top", "num_format": number_formats.get(kind, "General")})
            for kind in ("text", "integer", "ms", "us", "pct")
        }
        for band, color in PERCENTILE_COLORS.items()
    }
    snapshot_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        for candidate in row.get("_decision_workers", []):
            if isinstance(candidate, dict):
                snapshot_rows.append((row, candidate))
    snapshot_rows.sort(key=lambda item: (
        item[0].get("request_time_ms") is None,
        item[0].get("request_time_ms") or 0,
        item[0]["request_id"],
        as_number(item[1].get("estimatedTtftRank")) or 9_999_999,
    ))
    for output_row, (row, candidate) in enumerate(snapshot_rows, start=4):
        row_formats = band_formats[row["prefill_ttft_band"]]
        for col, (_, _, kind, getter) in enumerate(headers):
            write_cell(worksheet, output_row, col, getter(row, candidate), row_formats[kind])
        worksheet.set_row(output_row, 24)
    if snapshot_rows:
        worksheet.autofilter(3, 0, 3 + len(snapshot_rows), last_col)


def write_host_summary_sheet(workbook: xlsxwriter.Workbook, rows: list[dict[str, Any]]) -> None:
    worksheet = workbook.add_worksheet("Host Summary")
    fmt = formats(workbook)
    headers = [
        ("prefill_host", "prefill_host", "summary_text"),
        ("request_count", "request_count", "summary_int"),
        ("ttft_available", "TTFT available", "summary_int"),
        ("p50", "engine TTFT P50 ms", "summary_ms"),
        ("p90", "engine TTFT P90 ms", "summary_ms"),
        ("p95", "engine TTFT P95 ms", "summary_ms"),
        ("p99", "engine TTFT P99 ms", "summary_ms"),
        ("p99_tail_count", "global P99 tail count", "summary_int"),
        ("high_hit_tail_count", "high-hit P95+ count", "summary_int"),
        ("tail_input_queue_dominant", "tail input-queue dominant", "summary_int"),
        ("tail_scheduler_dominant", "tail scheduler dominant", "summary_int"),
        ("tail_running_dominant", "tail RUNNING→first dominant", "summary_int"),
        ("tail_with_low_hit_prior", "tail with earlier low-hit before first token", "summary_int"),
        ("cache_leader_count", "CACHE_LEADER", "summary_int"),
        ("shortest_ttft_count", "SHORTEST_TTFT", "summary_int"),
        ("shortest_ttft_low_cache_hit_count", "SHORTEST_TTFT_LOW_CACHE_HIT", "summary_int"),
    ]
    worksheet.hide_gridlines(2)
    fill_row(worksheet, 0, len(headers) - 1, fmt["title"])
    worksheet.write(0, 0, "Prefill-host summary (engine TTFT / WorkerStatus)", fmt["title"])
    fill_row(worksheet, 1, len(headers) - 1, fmt["note"])
    worksheet.write(1, 0, "Percentile basis", fmt["note"])
    worksheet.write(1, 1, "Per-host percentiles are calculated on prefill_engine_ttft_ms only. They do not represent Chat/Decode endpoint E2E.", fmt["note"])
    for col, (_, header, _) in enumerate(headers):
        worksheet.write(3, col, header, fmt["summary_header"])
        worksheet.set_column(col, col, max(18, len(header) + 3))
    worksheet.set_row(3, 34)
    worksheet.freeze_panes(4, 1)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prefill_host"]].append(row)
    for out_row, host in enumerate(sorted(grouped), start=4):
        host_rows = grouped[host]
        values = [float(row["prefill_engine_ttft_ms"]) for row in host_rows if row.get("prefill_engine_ttft_ms") is not None]
        tail_rows = [row for row in host_rows if row["prefill_ttft_band"] in {"P95-P99", "P99-P100"}]
        summary = {
            "prefill_host": host,
            "request_count": len(host_rows),
            "ttft_available": len(values),
            "p50": nearest_rank(values, 0.50),
            "p90": nearest_rank(values, 0.90),
            "p95": nearest_rank(values, 0.95),
            "p99": nearest_rank(values, 0.99),
            "p99_tail_count": sum(row["prefill_ttft_band"] == "P99-P100" for row in host_rows),
            "high_hit_tail_count": sum(bool(row["high_hit_tail_flag"]) for row in host_rows),
            "tail_input_queue_dominant": sum(row["dominant_observed_phase"] == "INPUT_QUEUE" for row in tail_rows),
            "tail_scheduler_dominant": sum("SCHEDULER" in row["dominant_observed_phase"] or "REMOTE_KV" in row["dominant_observed_phase"] for row in tail_rows),
            "tail_running_dominant": sum(row["dominant_observed_phase"] == "RUNNING_TO_FIRST_TOKEN" for row in tail_rows),
            "tail_with_low_hit_prior": sum(row["prior_low_hit_before_first_token_count"] > 0 for row in tail_rows),
            "cache_leader_count": sum(row["selection_reason"] == "CACHE_LEADER" for row in host_rows),
            "shortest_ttft_count": sum(row["selection_reason"] == "SHORTEST_TTFT" for row in host_rows),
            "shortest_ttft_low_cache_hit_count": sum(row["selection_reason"] == "SHORTEST_TTFT_LOW_CACHE_HIT" for row in host_rows),
        }
        for col, (key, _, format_key) in enumerate(headers):
            write_cell(worksheet, out_row, col, summary[key], fmt[format_key])
        worksheet.set_row(out_row, 22)
    worksheet.autofilter(3, 0, 3 + len(grouped), len(headers) - 1)


def write_data_scope_sheet(workbook: xlsxwriter.Workbook, sources: Sequence[PvSource],
                           rows: list[dict[str, Any]], event_counts: Counter[str],
                           selection_counts: Counter[str], threshold: dict[str, float | None],
                           start: datetime | str | None, end: datetime | str | None) -> None:
    worksheet = workbook.add_worksheet("Data Scope")
    fmt = formats(workbook)
    worksheet.hide_gridlines(2)
    worksheet.set_zoom(95)
    worksheet.set_column("A:A", 34)
    worksheet.set_column("B:B", 112)
    worksheet.write("A1", "Workbook scope and field interpretation", fmt["title"])
    worksheet.write_blank("B1", None, fmt["title"])
    start_time = _normalize_boundary(start)
    end_time = _normalize_boundary(end)
    entries = [
        ("Sources", "\n".join(f"{source.instance}: {source.path}" for source in sources)),
        ("Route decision window",
         f"[{start_time.isoformat() if start_time else '-∞'}, "
         f"{end_time.isoformat() if end_time else '+∞'}) in Asia/Shanghai; "
         "filter key is route.requestTimeMs. Cache/WorkerStatus events may come from the collector tail window."),
        ("PV records parsed", f"route={event_counts['route']:,}; cache_hit_comparison={event_counts['cache_hit_comparison']:,}; prefill_worker_status={event_counts['prefill_worker_status']:,}"),
        ("Request IDs in sheet", f"{len(rows):,}; rows with routing PV={sum(row['route_response_code'] is not None for row in rows):,}; complete joins={sum(row['telemetry_status'] == 'COMPLETE' for row in rows):,}"),
        ("What is not in this input", "Chat/Decode result, Chat HTTP outcome, Chat E2E duration, Prefill finish reason, and Prefill finish duration. The old chat_pd_* columns are deliberately not fabricated."),
        ("Percentile basis", "prefill_engine_ttft_ms = firstTokenTimeMs − inputQueueEnqueueTimeMs. Global nearest-rank thresholds: " + "; ".join(f"{key}={value:,.0f} ms" for key, value in threshold.items() if value is not None)),
        ("Route-to-first-token", "route_to_first_token_ms = firstTokenTimeMs − FlexLB requestTimeMs. It includes the pre-engine handoff and is kept beside engine TTFT for comparison."),
        ("Input queue", "input_queue_wait_ms is the WorkerStatus timing between its input-queue enqueue and drain timestamps."),
        ("Scheduler / Remote KV", "scheduler_to_running_ms is the combined time from scheduler handling to RUNNING. In the joined records, scheduler_to_running_ms = scheduler_wait_ms + remote_kv_wait_ms when all three values are valid. remote_kv_wait_ms is therefore a scheduler-to-running subphase, not an additional top-level duration."),
        ("RUNNING to first token", "running_to_first_token_ms measures time after the request reached RUNNING until first token. A high value can point to chunked-prefill / batch compute contention even if cache hit is high."),
        ("Cache tiers", "hbm_local_match_tokens + remote_kv_added_match_tokens = actual_hit_tokens for every complete joined record. This lets the table distinguish HBM-local hits from remote-KV supplied hits."),
        ("Same-host predecessor columns", "They count earlier route decisions on the same host whose first token timestamp is later than the current request's route decision time. low_hit_predecessor_* points to the most recent such request with actual hit rate < 90%. This is correlation evidence, not proof of queueing or causality."),
        ("Selection reasons", "; ".join(f"{reason}={count:,}" for reason, count in sorted(selection_counts.items()))),
        ("Decision snapshot", "The Decision Snapshot Top5 sheet is the top-five candidate set sampled by FlexLB at each route decision. Its queue/estimated-TTFT fields are routing token-work estimates, not observed latency. Selected snapshot fields are also at the left of Requests."),
        ("Row colors", "P0–P50 green; P50–P90 blue; P90–P95 yellow; P95–P99 orange; P99–P100 purple; missing TTFT / incomplete telemetry red."),
    ]
    for row, (key, value) in enumerate(entries, start=2):
        worksheet.write(row, 0, key, fmt["summary_header"])
        worksheet.write(row, 1, value, fmt["note"])
        worksheet.set_row(row, 34 if key in {"Sources", "Route decision window", "What is not in this input", "Scheduler / Remote KV", "RUNNING to first token", "Same-host predecessor columns"} else 24)


def build_workbook(sources: str | Path | PvSource | Sequence[str | Path | PvSource],
                   destination: str | Path, start: datetime | str | None = None,
                   end: datetime | str | None = None) -> dict[str, Any]:
    """Build the normalized analysis workbook and return an audit summary.

    ``sources`` accepts one raw log/CSV, a collector directory, explicit
    :class:`PvSource` objects, or any mixture of those.  Naive ``start``/``end``
    values are interpreted in Asia/Shanghai.  The end boundary is exclusive.
    """

    normalized_sources = _coerce_sources(sources)
    destination_path = Path(destination).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    rows, event_counts, selection_counts = build_rows(normalized_sources, start=start, end=end)
    if not rows:
        raise ValueError(
            "No routing PV records matched route.requestTimeMs in the requested "
            f"window; parsed route records={event_counts['route']:,}, "
            f"sources={len(normalized_sources):,}"
        )
    engine_ttft_values = [float(row["prefill_engine_ttft_ms"]) for row in rows
                          if row.get("prefill_engine_ttft_ms") is not None and row.get("route_success")]
    threshold = {
        "P50": nearest_rank(engine_ttft_values, 0.50),
        "P90": nearest_rank(engine_ttft_values, 0.90),
        "P95": nearest_rank(engine_ttft_values, 0.95),
        "P99": nearest_rank(engine_ttft_values, 0.99),
    }
    workbook = xlsxwriter.Workbook(destination_path)
    workbook.set_properties({
        "title": "FlexLB Prefill WorkerStatus P99 analysis",
        "subject": "PV route/cache/WorkerStatus request correlation",
        "author": "Codex",
        "comments": "Generated from raw FlexLB pv.log SLS export; no Chat/Decode E2E was available in input.",
    })
    write_requests_sheet(workbook, rows, threshold)
    write_p99_focus_sheet(workbook, rows, threshold)
    write_decision_snapshot_sheet(workbook, rows)
    write_host_summary_sheet(workbook, rows)
    write_data_scope_sheet(workbook, normalized_sources, rows, event_counts, selection_counts,
                           threshold, start, end)
    workbook.close()
    return {
        "destination": str(destination_path.resolve()),
        "source_count": len(normalized_sources),
        "sources": [
            {"instance": source.instance, "path": str(source.path)}
            for source in normalized_sources
        ],
        "window": {
            "start": _normalize_boundary(start).isoformat() if start is not None else None,
            "end": _normalize_boundary(end).isoformat() if end is not None else None,
            "semantics": "route.requestTimeMs in [start, end)",
        },
        "request_count": len(rows),
        "complete_request_count": sum(row["telemetry_status"] == "COMPLETE" for row in rows),
        "instance_count": len({row["flexlb_instance"] for row in rows}),
        "prefill_host_count": len({row["prefill_host"] for row in rows}),
        "event_counts": dict(event_counts),
        "selection_counts": dict(selection_counts),
        "thresholds_ms": threshold,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="+", help="raw pv.log/CSV file or collection directory")
    parser.add_argument("--output", "-o", required=True, help="destination .xlsx")
    parser.add_argument("--start", help="route decision start, ISO-8601; naive values use Asia/Shanghai")
    parser.add_argument("--end", help="route decision end (exclusive), ISO-8601")
    arguments = parser.parse_args()
    print(json.dumps(build_workbook(arguments.source, arguments.output,
                                    start=arguments.start, end=arguments.end),
                     ensure_ascii=False, indent=2))
