"""Persist startup phase costs using the caller's execution clock."""

from contextlib import contextmanager
from pathlib import Path
from artifacts.json_io import write_json


@contextmanager
def startup_phase(directory, phase, deadline):
    """Record preparation costs without introducing another execution clock."""
    path = Path(directory) / 'startup.json'
    started = deadline.clock()
    status, error = 'PASS', None
    try:
        deadline.check()
        yield
        deadline.check()
    except BaseException as exc:
        status, error = 'ERROR', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        import json
        rows = json.loads(path.read_text()) if path.exists() else []
        rows.append(dict(phase=phase, status=status, duration_s=deadline.clock()-started, error=error))
        write_json(path, rows)
