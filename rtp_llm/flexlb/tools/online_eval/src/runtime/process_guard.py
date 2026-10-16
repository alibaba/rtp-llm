"""Own a child session until completion or loss of the parent's pipe.

The guard shares its session with every workload descendant. Parent death closes
the only writer, so this process can reclaim the group without parent finally.
Deliberately escaped sessions still require the CI executor's container/cgroup.
"""

import os
import select
import signal
import subprocess
import sys


def main():
    if os.getpgrp() != os.getpid():
        raise RuntimeError("process guard requires its own session")
    descriptor = int(sys.argv[1])
    # Group TERM/INT reaches the workload too; the guard stays alive while its
    # child runs the separately budgeted cooperative cleanup.
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT, lambda *_: None)
    child = subprocess.Popen(sys.argv[2:])
    try:
        while child.poll() is None:
            readable, _, _ = select.select([descriptor], [], [], 0.1)
            if readable and not os.read(descriptor, 1):
                os.killpg(os.getpgrp(), signal.SIGKILL)
        code = child.wait()
        return code if code >= 0 else 128 - code
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
