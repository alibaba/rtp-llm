"""Atomic JSON checkpoints shared by execution and evidence producers."""

import json
from pathlib import Path
from artifacts.file_io import atomic_text


def write_json(path, payload, *, indent=None):
    path = Path(path)
    with atomic_text(path) as stream:
        json.dump(payload, stream, indent=indent, allow_nan=False,
                  separators=(",", ":") if indent is None else None)
        if indent is not None:
            stream.write("\n")
    return path
