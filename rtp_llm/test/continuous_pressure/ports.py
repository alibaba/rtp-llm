"""Contracts for the request transport, pressure polling and event output."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class RequestHandle(Protocol):
    def done(self) -> bool: ...

    def result(self) -> dict: ...

    def cancel(self) -> bool: ...


@dataclass(frozen=True)
class PressurePoll:
    samples: tuple
    raw: dict
    activity: tuple = ()
    step_page_full: bool = False
    step_history_complete: bool = True
    sample_complete: bool = True


class EventSink(Protocol):
    def __call__(self, event: dict, /) -> None: ...


class JsonlSink:
    """Append one event at a time to `root/events.jsonl` using bounded memory."""

    def __init__(self, root: str | Path):
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "events.jsonl"
        self._lock = threading.Lock()
        self._stream = self.path.open("a", encoding="utf-8")

    def __call__(self, event: dict) -> None:
        line = json.dumps(event, allow_nan=False) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
