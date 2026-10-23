"""Atomic UTF-8 text publication shared by reports and artifact rendering."""
from pathlib import Path
from contextlib import contextmanager
import tempfile


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
