"""Process-local AWI observation recorder.

Only real call sites in the serving path write here: the pretokenized
pipeline records prefill/decode windows from its own yields, and the KV
reuse counter comes from the scheduler's ``aux_info.reuse_len``.  No event
is synthesized from a requested setting.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional


_NETWORK_FETCHES = 0
_NETWORK_LOCK = threading.Lock()
_NETWORK_PATCHED = False


def install_network_counter() -> None:
    """Count real outbound TCP connects made by this process."""
    global _NETWORK_PATCHED
    with _NETWORK_LOCK:
        if _NETWORK_PATCHED:
            return
        _NETWORK_PATCHED = True
    import socket

    original = socket.socket.connect

    import ipaddress

    local_addresses = {"127.0.0.1", "::1", "localhost"}
    try:
        local_addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass

    def counted_connect(self, address):
        global _NETWORK_FETCHES
        try:
            host = address[0]
            remote = ipaddress.ip_address(host)
            is_local = (
                host in local_addresses
                or remote.is_loopback
                or remote.is_private
                or remote.is_link_local
            )
            if not is_local:
                with _NETWORK_LOCK:
                    _NETWORK_FETCHES += 1
        except Exception:
            pass
        return original(self, address)

    socket.socket.connect = counted_connect


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

    def load_weight_census(self) -> List[Dict[str, Any]]:
        """Merge the per-rank census files written during checkpoint load."""
        by_name: Dict[str, Dict[str, Any]] = {}
        for path in sorted(glob.glob("/tmp/awi/weights-*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        by_name[str(entry["name"])] = entry
            except (OSError, ValueError):
                continue
        with self._lock:
            for entry in self._weights:
                by_name[str(entry["name"])] = entry
        return [by_name[name] for name in sorted(by_name)]

    def observations(self, operation_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            events = list(self._events)
            op = self._operations.get(operation_id or "")
        weights = self.load_weight_census()
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
            "startup_network_fetches": _NETWORK_FETCHES,
            "weights": weights,
            "events": events,
        }


_TELEMETRY = AwiTelemetry()


def get_telemetry() -> AwiTelemetry:
    return _TELEMETRY


_WEIGHT_LOCK = threading.Lock()
_WEIGHT_DIR = os.environ.get("AWI_WEIGHT_CENSUS_DIR", "/tmp/awi")


def record_weight_placement(name: str, tensor: Any) -> None:
    """Record the actual storage of one loaded checkpoint tensor.

    Called by the safetensors iterator for every yielded key, so the census
    reflects the device the loader really used, including CPU/pinned Engram
    tables and any transformed layout.
    """
    try:
        device = tensor.device
        device_type = device.type
        index = device.index
        if device_type == "cuda" and index is None:
            try:
                index = torch.cuda.current_device()
            except Exception:
                index = 0
        if device_type == "cuda":
            device_name = "cuda:%d" % int(index)
            storage = "device"
            persistent = int(tensor.numel() * tensor.element_size())
        elif device_type == "cpu":
            device_name = "cpu"
            storage = "pinned" if bool(tensor.is_pinned()) else "anonymous"
            persistent = 0
        else:
            device_name = str(device)
            storage = str(device_type)
            persistent = 0
        entry = {
            "name": str(name),
            "device": device_name,
            "storage": storage,
            "persistent_gpu_bytes": persistent,
        }
    except Exception:
        return
    os.makedirs(_WEIGHT_DIR, exist_ok=True)
    rank = (
        os.environ.get("AWI_RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("RANK")
        or "0"
    )
    path = os.path.join(_WEIGHT_DIR, "weights-%s.jsonl" % rank)
    with _WEIGHT_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")


import torch  # noqa: E402  (used by record_weight_placement)


install_network_counter()
