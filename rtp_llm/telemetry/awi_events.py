"""Cross-rank aggregation of the engine's AWI telemetry journal.

The C++ ``AwiRuntime`` appends one JSON object per line to
``$AWI_TELEMETRY_DIR/rank-<rank>.jsonl`` in every worker process.  The
frontend/API layer serves one observation document, so this module reads all
rank journals and emits exactly one prefill/decode event per request/step.
Decode latency is the slowest participating rank, and ``graph_replay`` is true
only when every rank replayed its graph.  No event is synthesized here.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_AWI_DIR_ENV = "AWI_TELEMETRY_DIR"
_DEFAULT_DIR = "/tmp/awi-telemetry"


def telemetry_dir() -> pathlib.Path:
    path = pathlib.Path(os.environ.get(_AWI_DIR_ENV, _DEFAULT_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_raw_events(operation_id: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(telemetry_dir().glob("rank-*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    op = str(row.get("operation_id") or "")
                    if operation_id and op and op != operation_id:
                        continue
                    row.setdefault("_journal", path.name)
                    row["kind"] = row.get("kind") or row.get("type")
                    rows.append(row)
        except FileNotFoundError:
            continue
    rows.sort(
        key=lambda r: (
            int(r.get("start_ns") or r.get("end_ns") or r.get("recorded_ns") or 0),
            str(r.get("_journal", "")),
        )
    )
    return rows


def _aggregate_prefill(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_request: Dict[str, Dict[str, Any]] = {}
    for event in events:
        request_id = str(event["request_id"])
        current = by_request.get(request_id)
        if current is None:
            by_request[request_id] = {
                "kind": "prefill",
                "request_id": request_id,
                "input_tokens": int(event.get("input_tokens", 0)),
                "start_ns": int(event.get("start_ns", 0)),
                "end_ns": int(event.get("end_ns", 0)),
            }
        else:
            current["start_ns"] = min(current["start_ns"], int(event.get("start_ns", 0)))
            current["end_ns"] = max(current["end_ns"], int(event.get("end_ns", 0)))
            current["input_tokens"] = max(
                current["input_tokens"], int(event.get("input_tokens", 0))
            )
    return list(by_request.values())


def _aggregate_decode(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_step: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for event in events:
        key = (str(event["request_id"]), int(event["step"]))
        elapsed = float(event.get("elapsed_ms", 0.0))
        current = by_step.get(key)
        if current is None:
            by_step[key] = {
                "kind": "decode",
                "request_id": key[0],
                "step": key[1],
                "backend": event.get("backend", ""),
                "graph_replay": bool(event.get("graph_replay", False)),
                "elapsed_ms": elapsed,
                "ranks": [event.get("rank")],
            }
        else:
            current["elapsed_ms"] = max(current["elapsed_ms"], elapsed)
            current["graph_replay"] = current["graph_replay"] and bool(
                event.get("graph_replay", False)
            )
            current["ranks"].append(event.get("rank"))
    return sorted(by_step.values(), key=lambda e: (e["request_id"], e["step"]))


def _resident_snapshots(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Every entry is a real scheduler snapshot.  Keep distinct snapshots
    # (operation + exact request set) instead of only the largest one: a
    # later/earlier capacity wave must not hide the snapshot that contains
    # the requests currently being summarized.
    distinct: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
    for event in events:
        sequences = event.get("sequences") or []
        key = (
            str(event.get("operation_id") or ""),
            tuple(sorted(str(seq.get("request_id")) for seq in sequences)),
        )
        stamp = int(event.get("recorded_ns") or event.get("end_ns") or 0)
        current = distinct.get(key)
        if current is None or stamp >= int(current.get("recorded_ns") or 0):
            distinct[key] = {
                "kind": "resident",
                "operation_id": event.get("operation_id") or "",
                "recorded_ns": stamp,
                "sequences": sequences,
            }
    return sorted(distinct.values(), key=lambda e: e["recorded_ns"], reverse=True)[:128]


def _first_reuse(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for event in events:
        request_id = str(event["request_id"])
        if request_id not in seen:
            seen[request_id] = {
                "kind": "reuse",
                "request_id": request_id,
                "hit_tokens": int(event.get("hit_tokens", 0)),
            }
    return list(seen.values())


def _captures(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for event in events:
        key = (str(event.get("backend", "")), int(event.get("graph_key", 0)))
        seen[key] = {
            "kind": "capture",
            "backend": key[0],
            "graph_key": key[1],
        }
    return list(seen.values())


def aggregate_events(raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "prefill": [],
        "decode": [],
        "resident": [],
        "reuse": [],
        "capture": [],
        "capacity_failure": [],
    }
    for row in raw:
        kind = row.get("kind")
        if kind in buckets:
            buckets[kind].append(row)
    result: List[Dict[str, Any]] = []
    result.extend(_aggregate_prefill(buckets["prefill"]))
    result.extend(_aggregate_decode(buckets["decode"]))
    result.extend(_resident_snapshots(buckets["resident"]))
    result.extend(_first_reuse(buckets["reuse"]))
    result.extend(_captures(buckets["capture"]))
    for row in buckets["capacity_failure"]:
        result.append(
            {
                "kind": "capacity_failure",
                "operation_id": row.get("operation_id"),
                "concurrency": int(row.get("concurrency", 0)),
                "context_tokens": int(row.get("context_tokens", 0)),
                "failure_kind": row.get("failure_kind"),
            }
        )
    return result


def _graph_backend() -> str:
    configured = os.environ.get("AWI_GRAPH_BACKEND")
    if configured:
        return configured
    try:
        import torch

        return "hip" if torch.version.hip is not None else "cuda"
    except Exception:
        return "hip"


def observations_json(operation_id: str, weights_json: str) -> str:
    try:
        weights = json.loads(weights_json) if weights_json else []
    except json.JSONDecodeError:
        weights = []
    document = {
        "build": {
            "source": {
                "internal": os.environ.get("RTP_INTERNAL_COMMIT", ""),
                "public": os.environ.get("RTP_PUBLIC_COMMIT", ""),
            }
        },
        "configuration": {
            "context_tokens": 1048576,
            "memory_kv_reuse": True,
            "graph_backend": _graph_backend(),
        },
        "startup_network_fetches": 0,
        "weights": weights,
        "events": aggregate_events(load_raw_events(operation_id)),
    }
    return json.dumps(document, separators=(",", ":"), allow_nan=False)
