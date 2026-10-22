"""Observe completed graph forwards in a smoke server's console log."""

import time
from pathlib import Path

_GRAPH_COMPLETED = "[PyWrappedModel] CUDA graph forward completed"


def graph_replay_count(log_path: str) -> int:
    path = Path(log_path)
    if not path.is_file():
        return 0
    return path.read_text(errors="replace").count(_GRAPH_COMPLETED)


def require_graph_replay(log_path: str, previous_count: int) -> None:
    """Require a completed graph forward after this request, allowing log flush."""
    deadline = time.monotonic() + 5
    while True:
        if graph_replay_count(log_path) > previous_count:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"Request produced no completed CUDA/HIP graph replay; "
                f"possible eager fallback. Server log: {log_path}"
            )
        time.sleep(0.05)
