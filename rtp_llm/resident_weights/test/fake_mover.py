"""Stand-in for mover.py without CUDA: copies each received FD's bytes into a new /dev/shm file.

FAKE_MOVER_FAIL=to_host|to_gpu makes that direction fail; FAKE_MOVER_DELAY_S delays the answer.
"""

import importlib.util
import os
import sys
import tempfile
import time

# Load store.py by path: importing the package would pull in torch.
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "store.py")
_spec = importlib.util.spec_from_file_location("resident_store", _path)
store = importlib.util.module_from_spec(_spec)
sys.modules["resident_store"] = store
_spec.loader.exec_module(store)


def _copy(fd: int) -> int:
    out, path = tempfile.mkstemp(dir="/dev/shm")
    os.unlink(path)
    os.write(out, os.pread(fd, os.fstat(fd).st_size, 0))
    return out


def main() -> None:
    import socket

    if os.environ.get("FAKE_MOVER_EXIT_EARLY") == "1":
        sys.exit(2)
    sock = socket.socket(fileno=int(os.environ["RTP_LLM_RESIDENT_MOVER_FD"]))
    job, fds = store.recv_message(sock)
    direction = "to_host" if job["to_host"] else "to_gpu"
    delay_dir = os.environ.get("FAKE_MOVER_DELAY_DIR")
    if delay_dir in (None, "", direction):
        time.sleep(float(os.environ.get("FAKE_MOVER_DELAY_S", "0")))
    if os.environ.get("FAKE_MOVER_HALF_FRAME") == "1":
        sock.sendall(store._HEADER.pack(4096))
        time.sleep(60)
        return
    if os.environ.get("FAKE_MOVER_FAIL") == direction:
        store.send_message(sock, {"error": f"injected {direction} failure"})
        sys.exit(1)
    out = [_copy(fd) for fd in fds]
    store.send_message(sock, {}, out)
    exit_delay = float(os.environ.get("FAKE_MOVER_EXIT_DELAY_S", "0"))
    if exit_delay > 0:
        time.sleep(exit_delay)


if __name__ == "__main__":
    main()
