"""Atomic JSON checkpoints shared by execution and evidence producers."""

import json
from pathlib import Path
import tempfile


def write_json(path, payload, *, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=indent, allow_nan=False,
                      separators=(",", ":") if indent is None else None)
            if indent is not None:
                stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
