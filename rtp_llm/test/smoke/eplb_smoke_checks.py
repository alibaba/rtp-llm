"""Verify real, completed expert migration before the final smoke request."""

import json
import re
from pathlib import Path

_APPLIED = re.compile(
    r"\[EPLB\] applied model_id=(\d+) layers=(\[[\d, ]+\]) update=(\d+) "
    r"changed_slots=(\[[\d, ]+\]) ep_rank=(\d+)"
)


def check_eplb_applied(log_dir: Path, ep_size: int, model_ids=(0,)) -> list:
    """Require nonzero migration records for each model from every EP rank.

    A completed group update has identical layers and changed_slots on every rank.
    stdout and worker logs may contain the same record; duplicates are harmless.
    """
    updates = {}
    for path in log_dir.rglob("*.log*"):
        if not path.is_file():
            continue
        for match in _APPLIED.finditer(path.read_text(errors="replace")):
            model, layers, update, changed, rank = match.groups()
            model, update, rank = int(model), int(update), int(rank)
            layers, changed = tuple(json.loads(layers)), tuple(json.loads(changed))
            if model not in model_ids:
                raise AssertionError(
                    f"Unexpected EPLB migration for model_id={model} in {path}"
                )
            if len(layers) != len(changed):
                raise AssertionError(f"EPLB layer/changed-slot counts differ in {path}")
            if any(changed):
                updates.setdefault((model, layers, update), {})[rank] = changed

    completed = [
        (key, ranks)
        for key, ranks in sorted(updates.items())
        if set(ranks) == set(range(ep_size)) and len(set(ranks.values())) == 1
    ]
    records = []
    for model_id in model_ids:
        model_updates = [entry for entry in completed if entry[0][0] == model_id]
        if not model_updates:
            raise AssertionError(
                f"No completed EPLB migration for model_id={model_id} on all "
                f"{ep_size} ranks in {log_dir}; observed updates: {updates}"
            )
        (model, layers, update), ranks = model_updates[0]
        records.append(
            {
                "model_id": model,
                "layer_ids": list(layers),
                "update": update,
                "changed_slots": list(ranks[0]),
                "ep_ranks": sorted(ranks),
            }
        )
    return records
