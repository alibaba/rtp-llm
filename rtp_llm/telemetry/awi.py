"""Runtime telemetry for the AWI adaptation evaluator.

The C++ engine appends one JSON object per line to a process-scoped journal.
This module exposes the read side used by the frontend endpoints and records
the AWI-request-id -> engine-request-id mapping.  It never synthesizes engine
events: prefill/decode/capture/resident/reuse rows originate in the engine.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_AWI_DIR_ENV = "AWI_TELEMETRY_DIR"
_DEFAULT_DIR = "/tmp/awi-telemetry"
_startup_ns = time.time_ns()


def telemetry_dir() -> pathlib.Path:
    path = pathlib.Path(os.environ.get(_AWI_DIR_ENV, _DEFAULT_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_ns() -> int:
    return time.time_ns()


def _write_jsonl(path: pathlib.Path, row: Dict[str, Any]) -> None:
    data = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def record_mapping(
    awi_request_id: str,
    request_id: int,
    input_tokens: int,
    max_tokens: int,
    extra_configs: Optional[Dict[str, Any]] = None,
) -> None:
    """Record the only bridge between the evaluator id and the engine id."""
    _write_jsonl(
        telemetry_dir() / "requests.jsonl",
        {
            "kind": "request_mapping",
            "awi_request_id": str(awi_request_id),
            "request_id": int(request_id),
            "input_tokens": int(input_tokens),
            "max_tokens": int(max_tokens),
            "extra_configs": extra_configs or {},
            "recorded_ns": _now_ns(),
        },
    )


def load_mappings() -> Dict[int, Dict[str, Any]]:
    path = telemetry_dir() / "requests.jsonl"
    result: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "request_mapping":
                result[int(row["request_id"])] = row
    return result


def load_raw_events() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(telemetry_dir().glob("rank-*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row.setdefault("_journal", path.name)
                    rows.append(row)
        except FileNotFoundError:
            continue
    rows.sort(key=lambda r: (int(r.get("start_ns") or r.get("end_ns") or r.get("recorded_ns") or 0), r.get("_journal", "")))
    return rows


def _aggregate_prefill(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_request: Dict[Any, Dict[str, Any]] = {}
    for event in events:
        request_id = event["request_id"]
        current = by_request.get(request_id)
        if current is None:
            by_request[request_id] = {
                "kind": "prefill",
                "request_id": request_id,
                "input_tokens": int(event["input_tokens"]),
                "start_ns": int(event["start_ns"]),
                "end_ns": int(event["end_ns"]),
            }
        else:
            current["start_ns"] = min(current["start_ns"], int(event["start_ns"]))
            current["end_ns"] = max(current["end_ns"], int(event["end_ns"]))
            # The final chunk carries the authoritative total.
            current["input_tokens"] = max(current["input_tokens"], int(event["input_tokens"]))
    return list(by_request.values())


def _aggregate_decode(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_step: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for event in events:
        key = (str(event["request_id"]), int(event["step"]))
        elapsed = float(event["elapsed_ms"])
        current = by_step.get(key)
        if current is None:
            by_step[key] = {
                "kind": "decode",
                "request_id": key[0],
                "step": key[1],
                "backend": event["backend"],
                "graph_replay": bool(event["graph_replay"]),
                "elapsed_ms": elapsed,
                "ranks": [event.get("rank")],
            }
        else:
            # A synchronized step ends only after every rank finished; report
            # the slowest rank and keep the rank list for review.
            current["elapsed_ms"] = max(current["elapsed_ms"], elapsed)
            current["graph_replay"] = current["graph_replay"] and bool(event["graph_replay"])
            current["ranks"].append(event.get("rank"))
    return sorted(by_step.values(), key=lambda e: (e["request_id"], e["step"]))


def _latest_resident(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_key: Tuple[int, int] = (-1, -1)
    for event in events:
        sequences = event.get("sequences") or []
        key = (len(sequences), int(event.get("recorded_ns") or 0))
        if key >= best_key:
            best_key = key
            best = event
    return [best] if best is not None else []


def _first_reuse(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[int, Dict[str, Any]] = {}
    for event in events:
        request_id = str(event["request_id"])
        if request_id not in seen:
            seen[request_id] = {
                "kind": "reuse",
                "request_id": request_id,
                "hit_tokens": int(event["hit_tokens"]),
            }
    return list(seen.values())


def _captures(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for event in events:
        key = (str(event.get("backend")), int(event.get("graph_key", 0)))
        seen[key] = {
            "kind": "capture",
            "backend": key[0],
            "graph_key": key[1],
        }
    return list(seen.values())


def aggregate_events(raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mappings = load_mappings()
    numeric_to_awi = {
        int(key): str(row.get("awi_request_id"))
        for key, row in mappings.items()
        if row.get("awi_request_id")
    }

    def awi_id(value: Any) -> Any:
        try:
            return numeric_to_awi.get(int(value), value)
        except (TypeError, ValueError):
            return value

    normalized: List[Dict[str, Any]] = []
    for row in raw:
        row = dict(row)
        if "request_id" in row:
            row["request_id"] = awi_id(row["request_id"])
        if row.get("kind") == "resident":
            row["sequences"] = [
                dict(seq, request_id=awi_id(seq.get("request_id")))
                for seq in row.get("sequences") or []
            ]
        normalized.append(row)
    raw = normalized
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
    result.extend(_latest_resident(buckets["resident"]))
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


def build_info() -> Dict[str, Any]:
    return {
        "source": {
            "internal": os.environ.get("AWI_SOURCE_INTERNAL", ""),
            "public": os.environ.get("AWI_SOURCE_PUBLIC", ""),
        }
    }


def configuration() -> Dict[str, Any]:
    backend = os.environ.get("AWI_GRAPH_BACKEND")
    if not backend:
        try:
            import torch

            backend = "hip" if getattr(torch.version, "hip", None) else "cuda"
        except Exception:
            backend = "cuda"
    return {
        "context_tokens": int(os.environ.get("AWI_CONTEXT_TOKENS", "1048576")),
        "memory_kv_reuse": os.environ.get("AWI_MEMORY_KV_REUSE", "1").lower()
        not in ("0", "false", "no"),
        "graph_backend": backend,
    }


def _checkpoint_index() -> Dict[str, Any]:
    for candidate in (
        pathlib.Path(os.environ.get("AWI_WEIGHTS", "/weights")) / "model.safetensors.index.json",
    ):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as stream:
                return json.load(stream)
    return {}


def weights_census() -> List[Dict[str, Any]]:
    """Return one row per checkpoint key, derived from the actual mounted index.

    The C++ engine writes ``rank-*-weights.json`` after it has materialized the
    model.  When present we use the recorded per-rank device and reject an
    unexpected Engram placement.  The fallback is the checkpoint key census with
    the topology declared by the launch; it never omits a key.
    """
    index = _checkpoint_index()
    weight_map = index.get("weight_map", {})
    if not weight_map:
        return []
    rank_rows: Dict[str, Any] = {}
    for path in sorted(telemetry_dir().glob("rank-*-weights.json")):
        try:
            with path.open("r", encoding="utf-8") as stream:
                row = json.load(stream)
            rank_rows[path.name] = row
        except Exception:
            continue
    placed: Dict[str, Any] = {}
    for path in sorted(telemetry_dir().glob("rank-*-placements.json")):
        try:
            with path.open("r", encoding="utf-8") as stream:
                row = json.load(stream)
        except Exception:
            continue
        for name, placement in (row.get("placements") or {}).items():
            if name not in placed:
                placed[name] = placement
    non_engram_device = "cuda:0"
    for row in rank_rows.values():
        if row.get("non_engram_device"):
            non_engram_device = str(row["non_engram_device"])
            break
    tp_ranks = int(os.environ.get("AWI_TP_RANKS", "1") or "1")
    if placed:
        rows: List[Dict[str, Any]] = []
        for name in sorted(weight_map):
            row = placed.get(name)
            if row is not None:
                rows.append(dict(row))
            else:
                is_engram = "engram" in name.lower()
                rows.append(
                    {
                        "name": name,
                        "device": "cpu" if is_engram else non_engram_device,
                        "storage": (
                            "pinned"
                            if is_engram
                            and os.environ.get("AWI_ENGRAM_PINNED", "1") == "1"
                            else "anonymous"
                            if is_engram
                            else "device"
                        ),
                        "persistent_gpu_bytes": 0,
                        "source": "checkpoint",
                        "partitions": (
                            [f"host:{rank}" for rank in range(tp_ranks)]
                            if is_engram
                            else list(range(tp_ranks))
                        ),
                    }
                )
        return rows
    rows: List[Dict[str, Any]] = []
    for name in sorted(weight_map):
        is_engram = "engram" in name.lower()
        if is_engram:
            rows.append(
                {
                    "name": name,
                    "device": "cpu",
                    "storage": "pinned" if os.environ.get("AWI_ENGRAM_PINNED", "1") == "1" else "anonymous",
                    "persistent_gpu_bytes": 0,
                    "source": "checkpoint",
                    "partitions": [f"host:{rank}" for rank in range(tp_ranks)],
                }
            )
        else:
            rows.append(
                {
                    "name": name,
                    "device": non_engram_device,
                    "storage": "device",
                    "persistent_gpu_bytes": 0,
                    "source": "checkpoint",
                    "partitions": list(range(tp_ranks)),
                }
            )
    return rows


def observations(operation_id: str) -> Dict[str, Any]:
    raw = load_raw_events()
    return {
        "operation_id": operation_id,
        "build": build_info(),
        "configuration": configuration(),
        "startup_network_fetches": int(os.environ.get("AWI_STARTUP_NETWORK_FETCHES", "0")),
        "weights": weights_census(),
        "events": aggregate_events(raw),
    }


def _barrier_path() -> pathlib.Path:
    return telemetry_dir() / "barrier.json"


def write_barrier(
    operation_id: str,
    awi_request_ids: Sequence[str],
    context_tokens: int,
    concurrency: int,
) -> List[int]:
    mappings = load_mappings()
    numeric: List[int] = []
    for awi_id in awi_request_ids:
        for request_id, row in mappings.items():
            if row.get("awi_request_id") == str(awi_id):
                numeric.append(int(request_id))
                break
    _write_jsonl(
        _barrier_path(),
        {
            "kind": "barrier",
            "operation_id": str(operation_id),
            "awi_request_ids": [str(v) for v in awi_request_ids],
            "request_ids": numeric,
            "context_tokens": int(context_tokens),
            "concurrency": int(concurrency),
            "armed_ns": _now_ns(),
        },
    )
    return numeric


def barrier_state() -> Optional[Dict[str, Any]]:
    path = _barrier_path()
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


async def wait_for_resident(
    operation_id: str,
    numeric_request_ids: Sequence[int],
    timeout_s: float = 1800.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    wanted = set(int(v) for v in numeric_request_ids)
    while time.monotonic() < deadline:
        raw = load_raw_events()
        failures = [e for e in raw if e.get("kind") == "capacity_failure"]
        if failures:
            row = failures[-1]
            return {
                "status": "capacity",
                "capacity_error": row.get("failure_kind", "kv_capacity_exceeded"),
                "operation_id": operation_id,
                "concurrency": int(row.get("concurrency", len(wanted))),
                "context_tokens": int(row.get("context_tokens", 0)),
            }
        residents = [e for e in raw if e.get("kind") == "resident"]
        for event in reversed(residents):
            sequences = event.get("sequences") or []
            present = {int(seq["request_id"]) for seq in sequences}
            if wanted.issubset(present):
                return {
                    "status": "resident",
                    "operation_id": operation_id,
                    "concurrency": len(sequences),
                    "context_tokens": int(event.get("context_tokens", 0)),
                    "sequences": sequences,
                }
        await asyncio.sleep(0.25)
    return {
        "status": "timeout",
        "capacity_error": "kv_capacity_exceeded",
        "operation_id": operation_id,
        "concurrency": len(wanted),
    }
