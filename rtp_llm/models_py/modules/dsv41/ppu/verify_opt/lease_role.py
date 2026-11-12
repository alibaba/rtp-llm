"""Bazel-owned adapter for the existing native P/D performance lifecycle."""

import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def main():
    action, wire = sys.argv[1:]
    config = json.loads(wire)
    bundle = Path(config["adapter_bundle"])
    for name, expected in config["bundle_sha256"].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"measurement adapter changed: {name}")
    command = [sys.executable, "-Bu", str(bundle / "alignment_driver.py"), action, wire]
    if action == "load":
        subprocess.run(command, check=True, timeout=config["timeout"])
        return
    if action != "serve":
        raise ValueError(f"unknown native role action {action}")
    # GPU_COUNT=0 keeps the outer Bazel launcher from acquiring a second
    # lease. The native serving framework acquires and validates all eight.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RTP_V41_OPT_VISIBLE_DEVICES"]
    stop = Path(config["bazel_stop_file"])
    token = config["bazel_stop_token"]
    stop.parent.mkdir(parents=True, exist_ok=True)
    if stop.exists():
        raise ValueError("stop receipt already exists before launch")
    receipt = stop.with_suffix(".owner.json")
    with receipt.open("x") as stream:
        json.dump({"pid": os.getpid(), "token": token, "command": command}, stream)
    child = subprocess.Popen(command, stdin=subprocess.PIPE)
    deadline = time.monotonic() + config["timeout"] + 1800
    status = None
    try:
        # A Bazel test has no inherited interactive stdin. The coordinator
        # supplies a task-owned stop receipt; the native adapter still owns
        # the GPU lease, readiness/preflight and graceful service teardown.
        while child.poll() is None:
            if stop.exists():
                if stop.read_text().strip() != token:
                    raise ValueError("stop receipt token differs from owner")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("native role did not receive its bounded stop receipt")
            select.select([], [], [], 1)
    finally:
        child.stdin.close()
        status = child.wait(timeout=320)
    if status:
        raise RuntimeError(f"native role adapter exit={status}")


if __name__ == "__main__":
    main()
