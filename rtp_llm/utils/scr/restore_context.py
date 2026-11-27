"""Fresh, process-local inputs shared by all fixups after one SCR barrier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RestoreContext:
    pod_ip: str
