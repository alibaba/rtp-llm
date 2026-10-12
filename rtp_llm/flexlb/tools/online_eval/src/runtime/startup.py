"""Startup waits share the caller's budget and report the failing phase."""

import time
from contextlib import contextmanager
from pathlib import Path
from runtime.deadline import Deadline
from artifacts.json_io import write_json


def startup_deadline(deadline, timeout_s):
    return deadline if deadline is not None else Deadline(time.monotonic() + timeout_s)


def wait_ready(deadline, phase, probe, *, process=None, interval_s=0.5, stable_s=0):
    since = None
    while True:
        try:
            deadline.check()
            if process is not None and not process.alive():
                raise RuntimeError(f"{phase}: process exited\n{process.tail_log()}")
            ready = probe()
            deadline.check()
            if process is not None and not process.alive():
                raise RuntimeError(f"{phase}: process exited\n{process.tail_log()}")
            now = deadline.clock()
            since = (now if since is None else since) if ready else None
            if since is not None and now - since >= stable_s:
                return ready
            deadline.sleep(interval_s)
        except TimeoutError as exc:
            raise TimeoutError(f"{phase}: startup budget expired") from exc


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


def wait_flow_started(control, identity, process, deadline):
    from adapters.flow_control import read_status
    def acknowledged():
        state = read_status(control, identity)
        if not process.alive():
            if state.get('state') != 'DRAINED' or process.proc.poll() != 0:
                raise RuntimeError('client exited before sending acknowledgement\n' + process.tail_log())
        return state if state.get('state') in {'SENDING', 'DRAINING', 'DRAINED', 'INCOMPLETE'} else None
    return wait_ready(deadline, 'client/sending acknowledgement', acknowledged,
                      interval_s=0.1)
