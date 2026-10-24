"""Atomic UTF-8 text publication shared by reports and artifact rendering."""
from pathlib import Path
from contextlib import contextmanager
import tempfile
import json


@contextmanager
def atomic_text(path, *, encoding=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_text(path, content):
    with atomic_text(path, encoding='utf-8') as stream:
        stream.write(content)


def write_json(path, payload, *, indent=None):
    path = Path(path)
    with atomic_text(path) as stream:
        json.dump(payload, stream, indent=indent, allow_nan=False,
                  separators=(",", ":") if indent is None else None)
        if indent is not None:
            stream.write("\n")
    return path


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)
