"""CI console progress for the exact selected IDs; independent of gate verdicts."""
from contextlib import contextmanager
import hashlib
import json
import threading
import time


def snapshot(root, identities):
    done, active = [], []
    for identity in identities:
        profile = identity.rsplit('::', 1)[-1]
        directory = 'instance-' + hashlib.sha256(identity.encode()).hexdigest()
        results = list((root / profile).glob('lane*/part*-yaml/instances/' + directory + '/result.json'))
        if results:
            try:
                done.append(identity + '=' + json.loads(results[0].read_text())['status'])
            except (OSError, ValueError, KeyError):
                active.append(identity + ' (result pending)')
        else:
            active.append(identity)
    return done, active


@contextmanager
def progress(root, identities, *, interval_s=30):
    stopped = threading.Event()

    def report():
        while not stopped.wait(interval_s):
            done, active = snapshot(root, identities)
            print('[progress]', time.strftime('%Y-%m-%dT%H:%M:%S'),
                  'done=' + str(len(done)) + '/' + str(len(identities)),
                  'results=' + str(done), 'pending=' + str(active), flush=True)

    reporter = threading.Thread(target=report, name='ci-progress', daemon=True)
    reporter.start()
    try:
        yield
    finally:
        stopped.set()
        reporter.join()
