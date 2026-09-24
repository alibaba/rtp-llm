"""Bounded, crash-resilient metadata for the latest MegaMoE prefill."""

from __future__ import annotations

import contextvars
import itertools
import json
import logging
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist


_MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
_CURRENT = contextvars.ContextVar("mega_moe_prefill_snapshot", default=None)
_SEQUENCE = itertools.count(1)


def _snapshot_path() -> Path:
    directory = Path(
        os.environ.get(
            "MEGA_MOE_SNAPSHOT_DIR", f"/tmp/mega_moe_snapshots_{os.getuid()}"
        )
    )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    host = socket.gethostname().replace("/", "_")
    return directory / f"mega_moe_prefill_{host}_rank{rank}.json"


class _Snapshot:
    def __init__(self):
        self.path = _snapshot_path()
        self.data = {
            "prefill_sequence": next(_SEQUENCE),
            "started_at_ns": time.time_ns(),
            "state": "running",
            "dropped_oldest_launches": 0,
            "launches": [],
        }
        self.enabled = True
        self.next_index = 0
        self._write()

    def _write(self):
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.data, default=repr, separators=(",", ":"))
            while len(payload.encode("utf-8")) > _MAX_SNAPSHOT_BYTES and self.data["launches"]:
                self.data["launches"].pop(0)
                self.data["dropped_oldest_launches"] += 1
                payload = json.dumps(self.data, default=repr, separators=(",", ":"))
            fd, temp_path = tempfile.mkstemp(prefix=".mega_moe_", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    output.write(payload)
                os.replace(temp_path, self.path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except OSError:
            self.enabled = False
            logging.exception("Could not write MegaMoE prefill snapshot %s", self.path)

    def record(self, inputs):
        if not self.enabled:
            return
        self.data["launches"].append(
            {"index": self.next_index, "time_ns": time.time_ns(), "inputs": inputs}
        )
        self.next_index += 1
        self._write()

    def finish(self):
        self.data["state"] = "python_forward_returned"
        self._write()


@contextmanager
def mega_moe_prefill_snapshot(is_prefill: bool):
    """Keep one atomic snapshot for the current prefill on each rank."""
    if not is_prefill or os.environ.get("MEGA_MOE_LOG_INPUTS") != "1":
        yield
        return
    snapshot = _Snapshot()
    token = _CURRENT.set(snapshot)
    try:
        yield
    except BaseException:
        snapshot.data["state"] = "python_forward_failed"
        snapshot._write()
        raise
    else:
        snapshot.finish()
    finally:
        _CURRENT.reset(token)


def mega_moe_snapshot_active() -> bool:
    return _CURRENT.get() is not None


def record_mega_moe_launch(inputs):
    snapshot = _CURRENT.get()
    if snapshot is not None:
        snapshot.record(inputs)
