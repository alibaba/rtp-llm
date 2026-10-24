"""Verify real, completed expert migration before the final smoke request."""

import re
from pathlib import Path

_APPLIED = re.compile(
    r"\[EPLB\] applied model_id=(\d+) layer_id=(\d+) update=(\d+) "
    r"changed_slots=(\d+) ep_rank=(\d+)"
)


def check_eplb_applied(log_dir: Path, ep_size: int) -> list:
    """Require target-only, nonzero migration records from every EP rank.

    A completed update has the same global changed_slots on every rank.
    stdout and worker logs may contain the same record; duplicates are harmless.
    """
    updates = {}
    for path in log_dir.rglob("*.log*"):
        if not path.is_file():
            continue
        for match in _APPLIED.finditer(path.read_text(errors="replace")):
            model, layer, update, changed, rank = map(int, match.groups())
            if model != 0:
                raise AssertionError(
                    f"Unexpected draft EPLB migration for model_id={model} in {path}"
                )
            if changed > 0:
                updates.setdefault((model, layer, update), {})[rank] = changed

    completed = [
        (key, ranks)
        for key, ranks in sorted(updates.items())
        if set(ranks) == set(range(ep_size)) and len(set(ranks.values())) == 1
    ]
    if not completed:
        raise AssertionError(
            f"No completed EPLB migration for model_id=0 on all "
            f"{ep_size} ranks in {log_dir}; observed updates: {updates}"
        )
    (model, layer, update), ranks = completed[0]
    return [
        {
            "model_id": model,
            "layer_id": layer,
            "update": update,
            "changed_slots": ranks[0],
            "ep_ranks": sorted(ranks),
        }
    ]
