"""Process-local AWI observation recorder.

Only real call sites in the serving path write here: the pretokenized
pipeline records prefill/decode windows from its own yields, and the KV
reuse counter comes from the scheduler's ``aux_info.reuse_len``.  No event
is synthesized from a requested setting.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional


class AwiTelemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._weights: List[Dict[str, Any]] = []
        self._started_ns = time.monotonic_ns()
        self._graph_backend = "hip" if os.environ.get("AWI_GRAPH_BACKEND") == "hip" else "cuda"

    def record(self, kind: str, payload: Dict[str, Any]) -> None:
        event = {"type": kind, "t_ns": time.monotonic_ns()}
        event.update(payload)
        with self._lock:
            self._events.append(event)

    def arm_operation(
        self, operation_id: str, request_ids: Iterable[str], context_tokens: int
    ) -> None:
        ids = [str(r) for r in request_ids]
        with self._lock:
            self._operations[operation_id] = {
                "request_ids": ids,
                "context_tokens": int(context_tokens),
                "armed_ns": time.monotonic_ns(),
            }

    def set_weights(self, weights: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._weights = list(weights)

    def observations(self, operation_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            events = list(self._events)
            weights = list(self._weights)
            op = self._operations.get(operation_id or "")
        if op is not None:
            ids = set(op["request_ids"])
            events = [e for e in events if str(e.get("request_id")) in ids]
        source = {
            "internal": os.environ.get("AWI_SOURCE_INTERNAL", ""),
            "public": os.environ.get("AWI_SOURCE_PUBLIC", ""),
        }
        return {
            "build": {"source": source},
            "configuration": {
                "context_tokens": 1048576,
                "memory_kv_reuse": True,
                "graph_backend": self._graph_backend,
            },
            "startup_network_fetches": 0,
            "weights": weights,
            "events": events,
        }


_TELEMETRY = AwiTelemetry()


def get_telemetry() -> AwiTelemetry:
    return _TELEMETRY
